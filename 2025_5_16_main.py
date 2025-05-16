import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import tempfile # For dummy csv and tflite model

# For Framework QAT
import tensorflow_model_optimization as tfmot
quantize_model = tfmot.quantization.keras.quantize_model


# --- Configuration ---
DUMMY_CSV_FILEPATH = os.path.join(tempfile.gettempdir(), "dummy_conductance.csv")

config = {
    "filepath": "/Users/ch/dataset/quant/Depression_voltage_arranged/V_D = -1.6 V", # Path to your conductance CSV file
    "mode": 3,                     # Conductance processing mode (1, 2, or 3)
    "clip": False,                 # Clip weights to min/max of G_mean +/- G_std
    "use_scaling": True,           # Scale conductance values
    "scaling_factor": 1e4,        # Factor for scaling G_mean and G_std
    "noise_scaling_factor": 1,   # Factor to scale G_std for noise injection
    "use_int_levels": True,       # Use integer levels instead of CSV
    "int_min" : -100,
    "int_max" : 100,
    "epochs": 20,                   # Number of training epochs
    "batch_size": 64
}

# --- Dummy CSV Creation for Demonstration ---
def create_dummy_csv(filepath, num_pulses=50, num_readings=10):
    if not os.path.exists(filepath):
        print(f"Creating dummy CSV at: {filepath}")
        data = []
        for i in range(num_pulses):
            pulse_value = i * 0.1
            readings = [pulse_value] + list(np.random.normal(loc=pulse_value*1000 + 500, scale=pulse_value*50 + 20, size=num_readings-1))
            data.append(readings)
        df = pd.DataFrame(data)
        df.to_csv(filepath, sep='\t', header=False, index=False)
        print("Dummy CSV created.")

if not config["use_int_levels"] and not os.path.exists(config["filepath"]):
    create_dummy_csv(config["filepath"])
elif not config["use_int_levels"] and os.path.exists(config["filepath"]):
    print(f"Using existing CSV file: {config['filepath']}")


# --- Conductance Aware Model Handler ---
class ConductanceManager:
    def __init__(self, filepath, mode=1, clip=False, use_scaling=False, scaling_factor=1,
                 noise_scaling_factor=0, use_int_levels=False, int_min=0, int_max=0):
        self.filepath = filepath
        self.mode = mode
        self.use_clip = clip
        self.use_scaling = use_scaling
        self.scaling_factor = scaling_factor
        self.noise_scaling_factor = noise_scaling_factor
        self.use_int_levels = use_int_levels
        self.int_min = int_min
        self.int_max = int_max
    
        self.G_raw = None # Raw conductance values from CSV
        self.G_mean = None
        self.G_std = None
        self.min_weight_theoretical = None # Theoretical min based on G_mean - G_std
        self.max_weight_theoretical = None # Theoretical max based on G_mean + G_std

        self._load_conductance()

    def _load_conductance(self):
        if self.use_int_levels:
            levels = np.arange(self.int_min, self.int_max + 1, dtype=float)
            G_mean = levels.copy() # Positive and negative levels
            if self.mode == 3: # Symmetric around zero
                 G_mean = np.concatenate([-np.flip(levels[levels > 0]), [0] if 0 not in levels else [], levels[levels > 0]])
                 G_mean = np.unique(G_mean) # Ensure unique sorted values
            
            G_std = np.zeros_like(G_mean) # No std for integer levels by default
                                          # Or you can define a small constant noise
            print(f"Using integer levels: {self.int_min} to {self.int_max}")

        else:
            try:
                df = pd.read_csv(self.filepath, sep='\t', header=None)
                G_raw_data = df.to_numpy()
                # Assuming first column is pulse, rest are readings
                G_means_raw = np.array([np.mean(row[1:]) for row in G_raw_data if len(row) > 1])
                G_stds_raw = np.array([np.std(row[1:]) for row in G_raw_data if len(row) > 1])

                if self.use_scaling:
                    G_means_raw *= self.scaling_factor
                    G_stds_raw *= self.scaling_factor
                
                G_mean = G_means_raw
                G_std = G_stds_raw

            except FileNotFoundError:
                print(f"Error: Conductance file not found at {self.filepath}")
                print("Please provide a valid CSV file or use integer levels.")
                raise
            except Exception as e:
                print(f"Error loading or processing CSV: {e}")
                raise


        # 가중치 변형 모드 (mode 3 is typically used for symmetric weights around 0)
        if self.mode == 1: # Use as is (positive values typically)
            pass
        elif self.mode == 2: # Shift to be non-negative
            G_mean -= np.min(G_mean)
            # G_std remains the same as it's about spread
        elif self.mode == 3: # Symmetric: concatenate negative, zero, positive
            if not self.use_int_levels: # int_levels handles this above
                # Ensure G_mean is sorted before making it symmetric if it's from CSV
                sorted_indices = np.argsort(G_mean)
                G_mean_sorted = G_mean[sorted_indices]
                G_std_sorted = G_std[sorted_indices]

                # Create symmetric version, ensure 0 is included once if G_mean_sorted doesn't naturally pass through 0
                # We assume original G_mean from CSV is positive or has a mix that we want to make symmetric
                positive_part = G_mean_sorted[G_mean_sorted > 0]
                negative_part = -np.flip(positive_part) # Make negative counterparts
                
                G_mean_final = np.concatenate((negative_part, [0.0] if 0.0 not in G_mean_sorted else [], positive_part))
                
                # For G_std, we need to map it correctly.
                # If original G_std corresponds to G_mean_sorted, then flip and concat.
                std_positive_part = G_std_sorted[G_mean_sorted > 0]
                std_for_negative_part = np.flip(std_positive_part) # Std dev is always positive
                
                G_std_final = np.concatenate((std_for_negative_part, [np.min(G_std_sorted) if G_std_sorted.size > 0 else 0.0], std_positive_part)) # std for 0 can be min_std or 0
                
                # Remove duplicates that might arise if original G_mean had 0
                unique_means, unique_indices = np.unique(G_mean_final, return_index=True)
                G_mean = unique_means
                G_std = G_std_final[unique_indices]


        self.G_mean = G_mean
        self.G_std = G_std
        
        if self.G_mean is not None and len(self.G_mean) > 0 :
            print(f"[Conductance Levels] Count: {len(self.G_mean)}, Unique: {len(np.unique(self.G_mean))}")
            print(f"[G_mean] Min: {np.min(self.G_mean):.2e}, Max: {np.max(self.G_mean):.2e}, Mean: {np.mean(self.G_mean):.2e}")
            if len(self.G_std[self.G_std > 0]) > 0 : # Avoid warning if all stds are zero
                print(f"[G_std > 0] Min: {np.min(self.G_std[self.G_std > 0]):.2e}, Max: {np.max(self.G_std):.2e}, Mean: {np.mean(self.G_std):.2e}")
            else:
                print(f"[G_std] All std values are zero or G_std is empty.")

            self.min_weight_theoretical = np.min(self.G_mean - self.noise_scaling_factor * self.G_std)
            self.max_weight_theoretical = np.max(self.G_mean + self.noise_scaling_factor * self.G_std)
        else:
            print("Warning: G_mean is None or empty after _load_conductance. Check CSV or int_levels.")
            # Set defaults to avoid errors later if G_mean is empty
            self.G_mean = np.array([0.0])
            self.G_std = np.array([0.0])
            self.min_weight_theoretical = 0.0
            self.max_weight_theoretical = 0.0


    def get_kernel_constraint(self):
        if self.use_clip and self.min_weight_theoretical is not None and self.max_weight_theoretical is not None:
            class ClipWeight(tf.keras.constraints.Constraint):
                def __init__(self, min_val, max_val):
                    self.min_val = tf.cast(min_val, tf.float32)
                    self.max_val = tf.cast(max_val, tf.float32)
                def __call__(self, w):
                    return tf.clip_by_value(w, self.min_val, self.max_val)
                def get_config(self):
                    return {'min_val': float(self.min_val.numpy()), 'max_val': float(self.max_val.numpy())} # Ensure serializable
            return ClipWeight(self.min_weight_theoretical, self.max_weight_theoretical)
        return None

    def _quantize_value_to_g_mean(self, w_scalar):
        if self.G_mean is None or len(self.G_mean) == 0: return w_scalar # Should not happen
        idx = np.argmin(np.abs(self.G_mean - w_scalar))
        return self.G_mean[idx]

    def map_weights_to_g_mean(self, model):
        """Maps model weights to the closest G_mean values. For PTQ (custom)."""
        if self.G_mean is None:
            print("Error: G_mean not loaded. Cannot map weights.")
            return
        for layer in model.layers:
            if isinstance(layer, Dense):
                W, b = layer.get_weights()
                W_mapped = np.vectorize(self._quantize_value_to_g_mean)(W)
                layer.set_weights([W_mapped, b])
            elif isinstance(layer, tfmot.quantization.keras.QuantizeWrapperV2): # For framework QAT models
                # This is more complex as weights are not directly accessible in the same way.
                # Usually, framework QAT models are converted to TFLite, then weights extracted if needed.
                # For direct manipulation within QAT, it's trickier.
                # This function is primarily for custom PTQ/QAT on standard Keras layers.
                print(f"Skipping QuantizeWrapperV2 layer: {layer.name} in map_weights_to_g_mean")


    def add_noise_to_weights(self, model):
        """Adds noise to model weights based on G_std. For PTQ (custom) after mapping."""
        if self.G_mean is None or self.G_std is None:
            print("Error: G_mean or G_std not loaded. Cannot add noise.")
            return
        
        for layer in model.layers:
            if isinstance(layer, Dense):
                weights, biases = layer.get_weights()
                weights_flat = weights.flatten()
                noisy_weights_flat = np.empty_like(weights_flat)

                for i, w_val in enumerate(weights_flat):
                    # Find the closest G_mean level for the current weight
                    # This assumes weights are already mapped to G_mean, or we map them now to find corresponding std
                    idx = np.argmin(np.abs(self.G_mean - w_val))
                    
                    # Get std for that level
                    sigma_i = self.noise_scaling_factor * self.G_std[idx]
                    
                    # Add Gaussian noise
                    noise = np.random.normal(0.0, sigma_i)
                    noisy_weights_flat[i] = w_val + noise # Add noise to the (mapped) weight
                
                weights_noisy = noisy_weights_flat.reshape(weights.shape)
                layer.set_weights([weights_noisy, biases])
            elif isinstance(layer, tfmot.quantization.keras.QuantizeWrapperV2):
                 print(f"Skipping QuantizeWrapperV2 layer: {layer.name} in add_noise_to_weights")


    def get_custom_qat_callback(self, model_to_modify, inject_noise_during_training=True):
        """Callback for custom QAT: maps to G_mean and optionally adds noise."""
        class CustomQATCallback(tf.keras.callbacks.Callback):
            def __init__(self, conductance_manager, target_model, inject_noise):
                super().__init__()
                self.cm = conductance_manager
                self.target_model = target_model
                self.inject_noise = inject_noise

            def on_epoch_end(self, epoch, logs=None): # Can be on_train_batch_end for finer control
                print(f"\nCustom QAT callback at end of epoch {epoch+1}:")
                print("Mapping weights to G_mean...")
                self.cm.map_weights_to_g_mean(self.target_model)
                if self.inject_noise:
                    print("Adding noise to weights...")
                    self.cm.add_noise_to_weights(self.target_model)
                # self.cm.check_weights_mapped(self.target_model) # Optional check
        
        return CustomQATCallback(self, model_to_modify, inject_noise_during_training)

    def check_weights_mapped(self, model, tolerance=1e-6):
        """Checks if weights in Dense layers are close to values in G_mean."""
        if self.G_mean is None: return
        print("\nChecking if model weights are mapped to G_mean levels:")
        all_mapped = True
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Dense):
                weights = layer.get_weights()[0].flatten()
                mismatched_count = 0
                for w in weights:
                    if not np.any(np.isclose(w, self.G_mean, rtol=0, atol=tolerance)):
                        mismatched_count += 1
                if mismatched_count > 0:
                    print(f"Layer {i} ({layer.name}): {mismatched_count}/{len(weights)} weights NOT mapped to G_mean.")
                    all_mapped = False
                else:
                    print(f"Layer {i} ({layer.name}): All {len(weights)} weights mapped to G_mean.")
        if all_mapped:
            print("All Dense layer weights appear to be mapped correctly.")
        else:
            print("Some Dense layer weights are not mapped to G_mean.")


# --- Model Definition ---
def create_mnist_model(kernel_constraint=None, input_shape=(28, 28)):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(128, activation="relu", kernel_initializer=initializer, kernel_constraint=kernel_constraint),
        Dense(64, activation="relu", kernel_initializer=initializer, kernel_constraint=kernel_constraint),
        Dense(10, activation="softmax", kernel_initializer=initializer, kernel_constraint=kernel_constraint)
    ])
    return model

# --- MNIST Data ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# For TFLite PTQ representative dataset
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
    # Model has only one input so each data point has one element.
    yield [input_value]

# --- Evaluation Helper ---
def evaluate_model(model, model_name, x_test_data, y_test_data, is_tflite=False, tflite_interpreter=None):
    print(f"\n--- Evaluating: {model_name} ---")
    if is_tflite:
        if tflite_interpreter is None:
            print("Error: TFLite interpreter not provided.")
            return 0.0, 0.0
        
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        
        input_dtype = input_details[0]['dtype']
        # 입력 텐서의 양자화 파라미터 (스케일, 제로포인트) 가져오기
        input_quant_params = input_details[0]['quantization_parameters']
        input_scale = input_quant_params['scales'][0] if len(input_quant_params['scales']) > 0 else 0.0
        input_zero_point = input_quant_params['zero_points'][0] if len(input_quant_params['zero_points']) > 0 else 0
        
        is_input_quantized = (input_dtype == np.int8 or input_dtype == np.uint8) and input_scale != 0.0

        if is_input_quantized:
            print(f"[INFO] {model_name}: Input requires {input_dtype}. Scale={input_scale:.4e}, ZeroPoint={input_zero_point}")
        
        # 출력 텐서의 양자화 파라미터 가져오기 (필요한 경우)
        output_dtype = output_details[0]['dtype']
        output_quant_params = output_details[0]['quantization_parameters']
        output_scale = output_quant_params['scales'][0] if len(output_quant_params['scales']) > 0 else 0.0
        output_zero_point = output_quant_params['zero_points'][0] if len(output_quant_params['zero_points']) > 0 else 0
        is_output_quantized = (output_dtype == np.int8 or output_dtype == np.uint8) and output_scale != 0.0

        if is_output_quantized:
             print(f"[INFO] {model_name}: Output is {output_dtype}. Scale={output_scale:.4e}, ZeroPoint={output_zero_point}")

        num_correct = 0
        num_total = len(x_test_data)

        for i in range(num_total):
            # 1. 원본 float32 이미지 준비 (0~1 정규화된 상태)
            float_image_batch = np.expand_dims(x_test_data[i], axis=0).astype(np.float32)
            
            # 2. 입력 데이터 양자화 (모델이 int8/uint8 입력을 기대하고, 스케일/제로포인트가 유효할 경우)
            if is_input_quantized:
                quantized_image_batch = (float_image_batch / input_scale) + input_zero_point
                # 반올림 및 타입 캐스팅, 값 범위 클램핑
                quantized_image_batch = np.round(quantized_image_batch)
                if input_dtype == np.int8:
                    quantized_image_batch = np.clip(quantized_image_batch, -128, 127)
                elif input_dtype == np.uint8:
                    quantized_image_batch = np.clip(quantized_image_batch, 0, 255)
                processed_input = quantized_image_batch.astype(input_dtype)
            else:
                # 모델이 float32 입력을 기대하거나 양자화 파라미터가 없는 경우
                processed_input = float_image_batch.astype(input_dtype)
            
            # 3. 모델 입력 형태에 맞게 reshape (필요시)
            # 이 부분은 이전 코드의 reshape 로직을 processed_input에 맞게 적용합니다.
            # MNIST (1, 28, 28) -> 모델 기대 형태 (예: 1, 28, 28, 1 또는 1, 784)
            final_input = processed_input
            expected_shape = tuple(input_details[0]['shape'])
            current_shape = processed_input.shape

            if len(expected_shape) == 4 and expected_shape[3] == 1 and len(current_shape) == 3: 
                 final_input = np.expand_dims(final_input, axis=-1)
            elif len(expected_shape) == 2 and expected_shape[0] == 1 and len(current_shape) > 1 and expected_shape[1] == np.prod(current_shape[1:]): 
                final_input = final_input.reshape(1, -1)
            
            tflite_interpreter.set_tensor(input_details[0]['index'], final_input)
            tflite_interpreter.invoke()
            
            # 4. 출력 데이터 가져오기 및 역양자화 (모델이 int8/uint8 출력을 생성하는 경우)
            output_data = tflite_interpreter.get_tensor(output_details[0]['index'])
            
            if is_output_quantized:
                # int8/uint8 출력을 float32로 역양자화
                output_data_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
            else:
                output_data_float = output_data # 이미 float32 출력이거나 양자화 정보 없음

            if np.argmax(output_data_float) == y_test_data[i]:
                num_correct += 1
        accuracy = num_correct / num_total
        print(f"Test Accuracy: {accuracy:.4f}")
        return 0.0, accuracy
    else: # Keras 모델 평가
        loss, accuracy = model.evaluate(x_test_data, y_test_data, verbose=0)
        print(f"Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")
        return loss, accuracy


def print_weight_statistics(model, name):
    print(f"\nWeight Statistics for {name}:")
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Dense): # Works for standard Dense layers
            weights = layer.get_weights()[0]
            print(f" Layer {i} ({layer.name}): Min Weight = {np.min(weights):.6f}, Max Weight = {np.max(weights):.6f}, Mean = {np.mean(weights):.6f}, Std = {np.std(weights):.6f}")
        elif isinstance(layer, tfmot.quantization.keras.QuantizeWrapperV2): # For QAT (framework) layers
            # Accessing weights from QuantizeWrapperV2 is different.
            # It's often easier to convert to a fully quantized model first.
            # For a quick check, we can try to get underlying layer weights if possible,
            # but these are 'quantize-aware' weights, not the final int8 weights.
            try:
                # This gets the dequantized weights used during training
                unquantized_weights = layer.trainable_weights[0].numpy() # Example for kernel
                print(f" Layer {i} ({layer.name} - QAT Wrapper): Min Weight (dequantized) = {np.min(unquantized_weights):.6f}, Max Weight = {np.max(unquantized_weights):.6f}")
            except:
                 print(f" Layer {i} ({layer.name} - QAT Wrapper): Could not easily extract comparable weight stats.")


def plot_weight_hist(model, title="weight distribution", bins=50, is_tflite=False, tflite_interpreter=None):
    w_all = []
    if is_tflite and tflite_interpreter:
        tensor_details = tflite_interpreter.get_tensor_details()
        for tensor in tensor_details:
            if 'weight' in tensor['name'] or 'kernel' in tensor['name']: # Common names for weights
                w_all.extend(tflite_interpreter.get_tensor(tensor['index']).flatten())
    else: # Keras model
        for layer in model.layers:
            layer_weights = []
            if isinstance(layer, tf.keras.layers.Dense):
                layer_weights = layer.get_weights()[0].flatten()
            elif isinstance(layer, tfmot.quantization.keras.QuantizeWrapperV2):
                # For TF MOT QAT layers, weights are stored differently.
                # This typically gets the float weights that are being 'fake quantized'.
                for weight_variable in layer.trainable_weights: # Check all trainable weights
                    if "kernel" in weight_variable.name and "quantize_layer" not in weight_variable.name:
                        layer_weights = weight_variable.numpy().flatten()
                        break
            if len(layer_weights) > 0:
                 w_all.extend(layer_weights)

    if not w_all:
        print(f"No weights found for histogram: {title}")
        return

    plt.figure(figsize=(8,5))
    plt.hist(np.array(w_all), bins=bins)
    plt.title(title)
    plt.xlabel("Weight Value")
    plt.ylabel("Count")
    plt.show(block=False)
    plt.pause(0.1) # Give time for plot to render


# --- Main Experiment ---
results = {}

# Instantiate ConductanceManager based on global config
# This will be used for "Quantization-2 (Custom)"
# If use_int_levels is True, it will use integer levels. Otherwise, CSV.
conductance_handler = ConductanceManager(
    filepath=config["filepath"],
    mode=config["mode"],
    clip=config["clip"],
    use_scaling=config["use_scaling"],
    scaling_factor=config["scaling_factor"],
    noise_scaling_factor=config["noise_scaling_factor"],
    use_int_levels=config["use_int_levels"],
    int_min=config["int_min"],
    int_max=config["int_max"]
)

# 1. Baseline Float32 Model
print("\n=== 1. Baseline Float32 Model ===")
model_baseline = create_mnist_model(kernel_constraint=conductance_handler.get_kernel_constraint())
model_baseline.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model_baseline.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_split=0.1, verbose=1)
loss, acc = evaluate_model(model_baseline, "Baseline Float32", x_test, y_test)
results["Baseline_Float32"] = {"loss": loss, "accuracy": acc}
print_weight_statistics(model_baseline, "Baseline Float32")
plot_weight_hist(model_baseline, "Baseline Float32 Weights", bins = 500)


# scaling factor의 최적값을 구하는 로직
# baseline 모델의 가중치 최대값과 전도도의 최대값이 같아지도록 scaling factor 설정
'''# === 2. Baseline 모델에서 최대 절대 가중치 추출 ===
max_abs_baseline_weight = 0.0
for layer in model_baseline.layers:
    if isinstance(layer, Dense):
        layer_weights = layer.get_weights() # [kernel, bias] 리스트
        if layer_weights: # 가중치가 있는 레이어인지 확인
            kernel_weights = layer_weights[0] # 첫 번째가 커널 가중치
            current_max_abs = np.max(np.abs(kernel_weights))
            if current_max_abs > max_abs_baseline_weight:
                max_abs_baseline_weight = current_max_abs
print(f"\n[INFO] Determined Max Absolute Baseline Weight: {max_abs_baseline_weight:.4f}")


# === 3. 스케일링되지 않은 G_mean 로드 및 최대 절대값 추출 ===
# 임시 ConductanceManager를 사용하여 스케일링되지 않고 mode만 적용된 G_mean 로드
temp_conductance_handler_for_g_mean = ConductanceManager(
    filepath=config["filepath"],
    mode=config["mode"],
    clip=False, # 여기서는 무관
    use_scaling=False, # 중요: 스케일링 없이 G_mean 로드
    scaling_factor=1.0, # use_scaling=False 이므로 이 값은 무시됨
    noise_scaling_factor=config["noise_scaling_factor"], # G_mean 최대값 계산에는 무관
    use_int_levels=config["use_int_levels"],
    int_min=config["int_min"],
    int_max=config["int_max"]
)

if temp_conductance_handler_for_g_mean.G_mean is None or len(temp_conductance_handler_for_g_mean.G_mean) == 0:
    # use_int_levels=True 이고 G_mean이 잘 생성되었다면 이 오류는 발생하지 않음
    # CSV 파일 문제일 경우 발생 가능
    raise ValueError("G_mean (unscaled) could not be loaded or is empty from CSV. Cannot determine dynamic scaling factor.")

# mode 처리가 끝난 G_mean의 최대 절대값
max_abs_G_mean_unscaled = np.max(np.abs(temp_conductance_handler_for_g_mean.G_mean))
print(f"[INFO] Determined Max Absolute Unscaled G_mean (after mode processing): {max_abs_G_mean_unscaled:.4e}")


# === 4. 동적 scaling_factor 계산 ===
if max_abs_G_mean_unscaled == 0:
    print("[WARNING] Max Absolute Unscaled G_mean is 0. Defaulting dynamic_scaling_factor to 1.0 to avoid division by zero.")
    dynamic_scaling_factor = 1.0
else:
    dynamic_scaling_factor = max_abs_baseline_weight / max_abs_G_mean_unscaled

# 계산된 동적 스케일링 팩터를 config에 업데이트 (결과 저장 및 로깅용)
config["calculated_dynamic_scaling_factor"] = dynamic_scaling_factor
config["original_max_abs_baseline_weight"] = float(max_abs_baseline_weight) # float으로 변환 (JSON 직렬화)
config["original_max_abs_G_mean_unscaled"] = float(max_abs_G_mean_unscaled) # float으로 변환

print(f"[INFO] Calculated Dynamic Scaling Factor: {dynamic_scaling_factor:.4e}")


# === 5. 계산된 동적 scaling_factor로 메인 ConductanceManager 생성 ===
# 이후 모든 Custom 모델은 이 conductance_handler를 사용
conductance_handler = ConductanceManager(
    filepath=config["filepath"],
    mode=config["mode"],
    clip=config["clip"], # 이제 사용자가 설정한 clip (True 또는 False)을 사용
    use_scaling=True,    # 반드시 True로 설정하여 dynamic_scaling_factor 적용
    scaling_factor=dynamic_scaling_factor,
    noise_scaling_factor=config["noise_scaling_factor"],
    use_int_levels=config["use_int_levels"],
    int_min=config["int_min"],
    int_max=config["int_max"]
)'''

# --- 양자화-1: 프레임워크 지원 양자화 ---

# 2.a PTQ (Framework - TensorFlow Lite int8)
print("\n=== 2.a Framework PTQ (TFLite int8) ===")
# Note: Framework PTQ typically converts a pre-trained float model to int8.
# We use the trained model_baseline.
converter = tf.lite.TFLiteConverter.from_keras_model(model_baseline)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # Ensure int8 quantization
converter.inference_input_type = tf.int8  # or tf.uint8, depends on model/hardware
converter.inference_output_type = tf.int8 # or tf.uint8

try:
    tflite_model_ptq = converter.convert()
    # Save the model to a temporary file to load with interpreter
    with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmpfile:
        tmpfile.write(tflite_model_ptq)
        tflite_model_ptq_path = tmpfile.name

    interpreter_ptq = tf.lite.Interpreter(model_path=tflite_model_ptq_path)
    interpreter_ptq.allocate_tensors()
    loss_ptq_fw, acc_ptq_fw = evaluate_model(None, "Framework PTQ (TFLite int8)", x_test, y_test, is_tflite=True, tflite_interpreter=interpreter_ptq)
    results["Framework_PTQ_int8"] = {"loss": loss_ptq_fw, "accuracy": acc_ptq_fw}
    plot_weight_hist(None, "Framework PTQ (TFLite int8) Weights", is_tflite=True, tflite_interpreter=interpreter_ptq)   
    os.remove(tflite_model_ptq_path) # Clean up temp file
except Exception as e:
    print(f"Framework PTQ (TFLite int8) failed: {e}")
    results["Framework_PTQ_int8"] = {"loss": -1, "accuracy": -1, "error": str(e)}


# 2.b QAT (Framework - TF MOT for int8)
print("\n=== 2.b Framework QAT (TF MOT for int8) ===")
# Create a new model for QAT
model_qat_framework_base = create_mnist_model() # Fresh model
# Apply QAT
q_aware_model = quantize_model(model_qat_framework_base)
q_aware_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
q_aware_model.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_split=0.1, verbose=1)

loss_qat_fw, acc_qat_fw = evaluate_model(q_aware_model, "Framework QAT (TF MOT, before TFLite conversion)", x_test, y_test)
results["Framework_QAT_TF MOT_Keras"] = {"loss": loss_qat_fw, "accuracy": acc_qat_fw}
print_weight_statistics(q_aware_model, "Framework QAT (TF MOT Keras)")
plot_weight_hist(q_aware_model, "Framework QAT (TF MOT Keras) Weights")

# Convert QAT model to TFLite (common practice)
try:
    converter_qat = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter_qat.optimizations = [tf.lite.Optimize.DEFAULT] # This will use the QAT info
    tflite_model_qat = converter_qat.convert()

    with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmpfile:
        tmpfile.write(tflite_model_qat)
        tflite_model_qat_path = tmpfile.name

    interpreter_qat_tflite = tf.lite.Interpreter(model_path=tflite_model_qat_path)
    interpreter_qat_tflite.allocate_tensors()
    loss_qat_tfl, acc_qat_tfl = evaluate_model(None, "Framework QAT (TFLite from TF MOT)", x_test, y_test, is_tflite=True, tflite_interpreter=interpreter_qat_tflite)
    results["Framework_QAT_TFLite"] = {"loss": loss_qat_tfl, "accuracy": acc_qat_tfl}
    plot_weight_hist(None, "Framework QAT (TFLite from TF MOT) Weights", is_tflite=True, tflite_interpreter=interpreter_qat_tflite, bins = 500)
    os.remove(tflite_model_qat_path) # Clean up temp file
except Exception as e:
    print(f"Framework QAT (TFLite conversion/evaluation) failed: {e}")
    results["Framework_QAT_TFLite"] = {"loss": -1, "accuracy": -1, "error": str(e)}


# --- 양자화-2: 사용자 정의 전도도/정수 레벨 매핑 ---

# === 3.a Custom PTQ (G_mean mapping ONLY) === 
print("\n=== 3.b Custom PTQ (G_mean mapping ONLY) ===")
model_ptq_g_mean_only = tf.keras.models.clone_model(model_baseline)
model_ptq_g_mean_only.set_weights(model_baseline.get_weights())
model_ptq_g_mean_only.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Mapping weights to G_mean (NO noise will be added)...")
conductance_handler.map_weights_to_g_mean(model_ptq_g_mean_only)

loss_ptq_gmo, acc_ptq_gmo = evaluate_model(model_ptq_g_mean_only, "Custom PTQ (G_mean ONLY)", x_test, y_test)
results["Custom_PTQ_G_mean_ONLY"] = {"loss": loss_ptq_gmo, "accuracy": acc_ptq_gmo}
print_weight_statistics(model_ptq_g_mean_only, "Custom PTQ (G_mean ONLY)")
print("Checking mapping for G_mean ONLY model (should be very precise):")
conductance_handler.check_weights_mapped(model_ptq_g_mean_only, tolerance=1e-7) # 노이즈가 없으므로 매우 작은 허용오차로 확인
plot_weight_hist(model_ptq_g_mean_only, f"Custom PTQ (G_mean ONLY) {'Int' if config['use_int_levels'] else 'CSV'} Weights")


# 3.b Custom PTQ (G_mean mapping + Noise)
print("\n=== 3.a Custom PTQ (G_mean mapping + Noise) ===")
model_ptq_custom = tf.keras.models.clone_model(model_baseline)
model_ptq_custom.set_weights(model_baseline.get_weights()) # Start from trained baseline
model_ptq_custom.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) # Compile after cloning

print("Mapping weights to G_mean...")
conductance_handler.map_weights_to_g_mean(model_ptq_custom)
print("Adding noise based on G_std...")
conductance_handler.add_noise_to_weights(model_ptq_custom) # Add noise after mapping

loss, acc = evaluate_model(model_ptq_custom, "Custom PTQ (G_mean + Noise)", x_test, y_test)
results["Custom_PTQ_G_mean_Noise"] = {"loss": loss, "accuracy": acc}
print_weight_statistics(model_ptq_custom, "Custom PTQ (G_mean + Noise)")
conductance_handler.check_weights_mapped(model_ptq_custom) # Check mapping (noise will shift them slightly)
plot_weight_hist(model_ptq_custom, f"Custom PTQ (G_mean + Noise) {'Int' if config['use_int_levels'] else 'CSV'} Weights")


# 3.c Custom QAT (G_mean mapping + Noise during training)
print("\n=== 3.b Custom QAT (G_mean mapping + Noise during training) ===")
model_qat_custom = create_mnist_model(kernel_constraint=conductance_handler.get_kernel_constraint())
model_qat_custom.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Custom QAT callback: True to inject noise during training
# For QAT, noise should be injected *during* training
custom_qat_callback = conductance_handler.get_custom_qat_callback(model_qat_custom, inject_noise_during_training=True)

model_qat_custom.fit(x_train, y_train,
                     epochs=config["epochs"],
                     batch_size=config["batch_size"],
                     validation_split=0.1,
                     callbacks=[custom_qat_callback],
                     verbose=1)

# After training with the callback, the model's weights should reflect the QAT process.
# Evaluate this model. Noise was already part of its training.
loss, acc = evaluate_model(model_qat_custom, "Custom QAT (G_mean + Noise in training)", x_test, y_test)
results["Custom_QAT_G_mean_Noise_In_Training"] = {"loss": loss, "accuracy": acc}
print_weight_statistics(model_qat_custom, "Custom QAT (G_mean + Noise in training)")
conductance_handler.check_weights_mapped(model_qat_custom) # Check if weights are still on G_mean (noise makes them deviate)
plot_weight_hist(model_qat_custom, f"Custom QAT (G_mean + Noise in training) {'Int' if config['use_int_levels'] else 'CSV'} Weights")


# --- Final Results Summary ---
print("\n\n--- Experiment Summary ---")
if config["use_int_levels"]:
    print(f"Mode: Integer Levels ({config['int_min']} to {config['int_max']})")
else:
    print(f"Mode: CSV Conductance Data (Path: {config['filepath']})")
print(f"Scaling: {config['use_scaling']}, Factor: {config['scaling_factor']}")
print(f"Noise Scaling Factor: {config['noise_scaling_factor']}")

for model_name, metrics in results.items():
    if "error" in metrics:
        print(f"{model_name}: Accuracy = {metrics.get('accuracy', 'N/A')}, Loss = {metrics.get('loss', 'N/A')} (Error: {metrics['error']})")
    else:
        print(f"{model_name}: Accuracy = {metrics['accuracy']:.4f}, Loss = {metrics['loss']:.4f}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"/Users/ch/python/quantization/results/mnist_quant_experiment_{timestamp}" # Current directory relative
os.makedirs(save_dir, exist_ok=True)

config_save_path = os.path.join(save_dir, "config.json")
results_save_path = os.path.join(save_dir, "results.json")

with open(config_save_path, "w") as f:
    # Convert numpy types in config if any (though current config is fine)
    serializable_config = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in config.items()}
    json.dump(serializable_config, f, indent=4)

with open(results_save_path, "w") as f:
    serializable_results = {}
    for k, v_dict in results.items():
        serializable_results[k] = {sub_k: (float(sub_v) if isinstance(sub_v, (np.floating, float)) else sub_v) for sub_k, sub_v in v_dict.items()}
    json.dump(serializable_results, f, indent=4)

print(f"\nConfiguration and results saved to: {save_dir}")
plt.show(block=True) # Keep plots open until manually closed
