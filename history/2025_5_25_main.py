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

np.random.seed(2)

config = {
    "filepath": "/Users/ch/dataset/quant/Depression_voltage_arranged/V_D = -1.6 V", # Path to your conductance CSV file
    "mode": 3,                     # Conductance processing mode (1, 2, or 3)
    "clip": False,                 # Clip weights to min/max of G_mean +/- G_std
    "use_scaling": True,           # Scale conductance values
    "scaling_factor": 1e4,         # Factor for scaling G_mean and G_std
    "noise_scaling_factor": 1,     # Factor to scale G_std for noise injection (0 for no noise during custom QAT/PTQ noise step)
    "num_of_G":16,                # Number of sampled conductance levels in G, 1~100, -1 = 100
    "G_sampling_method": "uniform_interval", # conductance sampling method, "uniform_interval" or "random"
    "epochs": 10,                  # Number of training epochs
    "batch_size": 64,
    "apply_on_batch": True,       # QAT적용 level, False: epoch, True: batch
    "use_int_levels": False,      # 삭제 예정
    "int_min": -100,              # 삭제 예정
    "int_max": 100                # 삭제 예정
}

np.random.seed(21342145)

# --- Conductance Aware Model Handler ---
class ConductanceManager:
    def __init__(self, filepath, mode=1, clip=False, use_scaling=False, scaling_factor=1,
                 noise_scaling_factor=0, use_int_levels=False, int_min=0, int_max=0, apply_on_batch=False,
                 num_of_G=-1, G_sampling_method="uniform_interval"):
        self.filepath = filepath
        self.mode = mode
        self.use_clip = clip
        self.use_scaling = use_scaling
        self.scaling_factor = scaling_factor
        self.noise_scaling_factor = noise_scaling_factor
        self.num_of_G = num_of_G if num_of_G != -1 else 100 # -1을 100으로 해석
        self.G_sampling_method = G_sampling_method
        self.apply_on_batch = apply_on_batch

        self.G_mean = None
        self.G_std = None
        self.min_weight = None
        self.max_weight = None
        self.use_int_levels = use_int_levels
        self.int_min = int_min
        self.int_max = int_max
        self._load_conductance()


    # 전도도를 다루는 함수
    def _load_conductance(self):
        G_mean_final_val = None
        G_std_final_val = None

        df = pd.read_csv(self.filepath, sep='\t', header=None)
        G_raw = df.to_numpy()
        if self.use_scaling:
            G_raw *= self.scaling_factor

        G_means_raw = np.array([np.mean(row[1:]) for row in G_raw])
        G_stds_raw = np.array([np.std(row[1:]) for row in G_raw])

        paired_values = sorted(zip(G_means_raw, G_stds_raw), key=lambda x: x[0])

        unique_G_means_map = {}
        for mean, std in paired_values:
            if mean not in unique_G_means_map: # 첫번째 값(가장 작은 std) 사용
                unique_G_means_map[mean] = std

        available_G_means = np.array(sorted(list(unique_G_means_map.keys())))
        available_G_stds = np.array([unique_G_means_map[mean] for mean in available_G_means])

        num_available_G_means = len(available_G_means)

        # 전도도 파일에서 사용할 전도도 값의 개수를 정하는 코드
        '''
        random: 전도도 값 중 무작위 추출 
        uniform_interval: 전도도 값 중 일정한 간격으로 추출
        '''
        if self.G_sampling_method == "random":
            random_indices = np.sort(np.random.choice(num_available_G_means, size=self.num_of_G, replace=False))
            selected_G_means = available_G_means[random_indices]
            selected_G_stds = available_G_stds[random_indices]
        elif self.G_sampling_method == "uniform_interval":
            if self.num_of_G == 1:
                indices_to_select = np.array([num_available_G_means // 2], dtype=int)
            else:
                indices_to_select = np.linspace(0, num_available_G_means - 1, num=self.num_of_G, dtype=int)
            selected_G_means = available_G_means[indices_to_select]
            selected_G_stds = available_G_stds[indices_to_select]
        
        print(f"Selected {len(selected_G_means)} positive conductance levels out of {num_available_G_means} available (Target: {self.num_of_G}, Method: {self.G_sampling_method if self.num_of_G < num_available_G_means and self.num_of_G > 0 else 'all'}).")

        # 전도도 선택 모드
        '''
        mode 1 : 전도도 값을 그대로 사용
        mode 2 : 전도도의 최솟값이 0이 되도록 평행이동
        mode 3 : 전도도 배열에 -1을 곱한 값, 0을 기존 전도도 배열에 추가
        '''
        if self.mode == 1:
            G_mean_final_val = selected_G_means
            G_std_final_val = selected_G_stds
        elif self.mode == 2:
            G_mean_final_val = selected_G_means - np.min(selected_G_means)
            G_std_final_val = selected_G_stds
        elif self.mode == 3:
            negative_part = -np.flip(selected_G_means)
            std_for_negative_part = np.flip(selected_G_stds)
            std_for_zero = np.min(selected_G_stds)

            G_mean_final_val = np.concatenate((negative_part, [0.0], selected_G_means))
            G_std_final_val = np.concatenate((std_for_negative_part, [std_for_zero], selected_G_stds))
            
            unique_means, unique_indices = np.unique(G_mean_final_val, return_index=True)
            G_mean_final_val = unique_means
            G_std_final_val = G_std_final_val[unique_indices]

        self.G_mean = G_mean_final_val
        self.G_std = G_std_final_val

    
        print(f"[Conductance Levels] Count: {len(self.G_mean)}, Unique: {len(np.unique(self.G_mean))}")
        print(f"[G_mean] Min: {np.min(self.G_mean):.2e}, Max: {np.max(self.G_mean):.2e}, Mean: {np.mean(self.G_mean):.2e}")
        print(f"[G_std > 0] Min: {np.min(self.G_std[self.G_std > 0]):.2e}, Max: {np.max(self.G_std):.2e}, Mean: {np.mean(self.G_std):.2e}")
        
        # 나중에 noise를 사용할 것을 반영해서 self.noise_scaling_factor를 계산해준다
        self.min_weight = np.min(self.G_mean - self.noise_scaling_factor * self.G_std)
        self.max_weight = np.max(self.G_mean + self.noise_scaling_factor * self.G_std)


    # 학습시 가중치 범위 clip
    def get_kernel_constraint(self):
        if self.use_clip:
            class ClipWeight(tf.keras.constraints.Constraint):
                def __init__(self, min_val, max_val):
                    self.min_val = tf.cast(min_val, tf.float32)
                    self.max_val = tf.cast(max_val, tf.float32)
                def __call__(self, w):
                    return tf.clip_by_value(w, self.min_val, self.max_val)
                def get_config(self):
                    return {'min_val': float(self.min_val.numpy()), 'max_val': float(self.max_val.numpy())}
            return ClipWeight(self.min_weight, self.max_weight)
        return None


    
    # G_mean - weight가 최솟값인 G_mean의 idx를 찾는 함수
    def _quantize_value_to_g_mean(self, w_scalar):
        idx = np.argmin(np.abs(self.G_mean - w_scalar))
        return self.G_mean[idx]


    # G_mean[idx]로 가중치를 mapping하는 함수
    def map_weights_to_g_mean(self, model):
        for layer in model.layers:
            if isinstance(layer, Dense):
                W, b = layer.get_weights()
                W_mapped = np.vectorize(self._quantize_value_to_g_mean)(W)
                layer.set_weights([W_mapped, b])
            elif isinstance(layer, tfmot.quantization.keras.QuantizeWrapperV2):
                print(f"Skipping QuantizeWrapperV2 layer: {layer.name} in map_weights_to_g_mean (expected for framework QAT)")


    # 가중치에 noise를 적용하는 함수
    def add_noise_to_weights(self, model):
        if self.noise_scaling_factor == 0:
            print("Noise scaling factor is 0, no noise will be added.")
            return

        for layer in model.layers:
            if isinstance(layer, Dense):
                weights, biases = layer.get_weights()
                weights_flat = weights.flatten()
                noisy_weights_flat = np.empty_like(weights_flat)

                for i, w_val in enumerate(weights_flat):
                    idx = np.argmin(np.abs(self.G_mean - w_val))
                    sigma_i = self.noise_scaling_factor * self.G_std[idx]
                    if sigma_i < 0: sigma_i = 0
                    noise = np.random.normal(0.0, sigma_i)
                    noisy_weights_flat[i] = w_val + noise
                
                weights_noisy = noisy_weights_flat.reshape(weights.shape)
                layer.set_weights([weights_noisy, biases])
            elif isinstance(layer, tfmot.quantization.keras.QuantizeWrapperV2):
                 print(f"Skipping QuantizeWrapperV2 layer: {layer.name} in add_noise_to_weights (expected for framework QAT)")


    # QAT callback 함수
    def get_custom_qat_callback(self, model_to_modify, config_param, inject_noise_during_training=True):
        apply_on_batch = config_param.get("apply_on_batch", False)
        class CustomQATCallback(tf.keras.callbacks.Callback):
            def __init__(self, conductance_manager, target_model, inject_noise, apply_on_batch_flag):
                super().__init__()
                self.cm = conductance_manager
                self.target_model = target_model
                self.inject_noise = inject_noise
                self.apply_on_batch = apply_on_batch_flag

            def _perform_qat_operations(self, context_str="Epoch"):
                print(f"\nCustom QAT callback in {context_str}:") # epoch 혹은 batch마다 qat적용 후 콜백
                print(f"  Mapping weights to G_mean...")          # epoch 혹은 batch마다 qat적용 후 콜백
                self.cm.map_weights_to_g_mean(self.target_model)
                if self.cm.noise_scaling_factor > 0:
                    print(f"  Adding noise to weights...")
                    self.cm.add_noise_to_weights(self.target_model)

            # --- 에포크 단위 콜백 ---
            def on_epoch_end(self, epoch, logs=None):
                if not self.apply_on_batch: # 배치 단위가 아닐 때만 에포크 단위 실행
                    self._perform_qat_operations(context_str=f"end of epoch {epoch+1}")

            # --- 배치 단위 콜백 ---
            def on_train_batch_end(self, batch, logs=None):
                if self.apply_on_batch: # 배치 단위일 때만 실행
                    # 모든 배치마다 실행하면 너무 빈번할 수 있으므로, 특정 N 배치마다 실행하도록 조절 가능
                    if batch % 100 == 0:
                        self._perform_qat_operations(context_str=f"end of batch {batch+1}")

        return CustomQATCallback(self, model_to_modify, inject_noise_during_training, apply_on_batch)
    

    def check_weights_mapped(self, model, tolerance=1e-6):
        if self.G_mean is None or self.G_mean.size == 0:
            print("\nChecking weights mapped: G_mean is not set or is empty. Cannot check mapping.")
            return

        print("\nChecking if model weights are mapped to G_mean levels:")
        all_mapped = True
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Dense):
                weights_data = layer.get_weights()
                if not weights_data : # No weights in layer
                    print(f"Layer {i} ({layer.name}): No weights found to check.")
                    continue
                weights = weights_data[0].flatten()
                mismatched_count = 0
                for w in weights:
                    # Check if w is close to any value in self.G_mean
                    if not np.any(np.isclose(w, self.G_mean, rtol=0, atol=tolerance)):
                        mismatched_count += 1
                if mismatched_count > 0:
                    print(f"Layer {i} ({layer.name}): {mismatched_count}/{len(weights)} weights NOT mapped (tolerance={tolerance:.1e}).")
                    all_mapped = False
                else:
                    print(f"Layer {i} ({layer.name}): All {len(weights)} weights mapped (tolerance={tolerance:.1e}).")
        if all_mapped: print("All Dense layer weights appear to be mapped correctly.")
        else: print("Some Dense layer weights are not mapped to G_mean (or outside tolerance).")

# --- Batch/Epoch Metrics Callback ---
class BatchMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, print_batch_loss_freq=0): # print_batch_loss_freq: 0=never, 1=every batch, N=every N batches
        super().__init__()
        self.batch_losses = []
        self.epoch_train_losses = []
        self.epoch_train_accuracies = []
        self.epoch_val_losses = []
        self.epoch_val_accuracies = []
        self.print_batch_loss_freq = print_batch_loss_freq

    def on_train_begin(self, logs=None):
        self.batch_losses = []
        self.epoch_train_losses = []
        self.epoch_train_accuracies = []
        self.epoch_val_losses = []
        self.epoch_val_accuracies = []
        print("Starting training...")

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))
        if self.print_batch_loss_freq > 0 and (batch + 1) % self.print_batch_loss_freq == 0:
            print(f" - Batch {batch+1}: loss = {logs.get('loss'):.4f}")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch+1}/{self.params['epochs']}")

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_train_losses.append(logs.get('loss'))
        self.epoch_train_accuracies.append(logs.get('accuracy'))
        self.epoch_val_losses.append(logs.get('val_loss'))
        self.epoch_val_accuracies.append(logs.get('val_accuracy'))
        print(f"Epoch {epoch+1}: loss={logs.get('loss'):.4f}, acc={logs.get('accuracy'):.4f}, val_loss={logs.get('val_loss'):.4f}, val_acc={logs.get('val_accuracy'):.4f}")


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


def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
    yield [input_value]


# --- Evaluation Helper ---
def evaluate_model(model, model_name, x_test_data, y_test_data, is_tflite=False, tflite_interpreter=None):
    print(f"\n--- Evaluating: {model_name} ---")
    if is_tflite:
        if tflite_interpreter is None:
            print("Error: TFLite interpreter not provided.")
            return 0.0, 0.0 # Loss, Accuracy

        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        input_dtype = input_details[0]['dtype']
        input_quant_params = input_details[0]['quantization_parameters']
        input_scale_list = input_quant_params['scales']
        input_zp_list = input_quant_params['zero_points']

        input_scale = input_scale_list[0] if len(input_scale_list) > 0 else 0.0
        input_zero_point = input_zp_list[0] if len(input_zp_list) > 0 else 0
        is_input_quantized = (input_dtype == np.int8 or input_dtype == np.uint8) and input_scale != 0.0

        output_dtype = output_details[0]['dtype']
        output_quant_params = output_details[0]['quantization_parameters']
        output_scale_list = output_quant_params['scales']
        output_zp_list = output_quant_params['zero_points']
        
        output_scale = output_scale_list[0] if len(output_scale_list) > 0 else 0.0
        output_zero_point = output_zp_list[0] if len(output_zp_list) > 0 else 0
        is_output_quantized = (output_dtype == np.int8 or output_dtype == np.uint8) and output_scale != 0.0
        
        if is_input_quantized: print(f"[INFO] {model_name}: Input requires {input_dtype}. Scale={input_scale:.4e}, ZeroPoint={input_zero_point}")
        if is_output_quantized: print(f"[INFO] {model_name}: Output is {output_dtype}. Scale={output_scale:.4e}, ZeroPoint={output_zero_point}")

        num_correct = 0
        num_total = len(x_test_data)
        for i in range(num_total):
            float_image_batch = np.expand_dims(x_test_data[i], axis=0).astype(np.float32)
            if is_input_quantized:
                quantized_image_batch = (float_image_batch / input_scale) + input_zero_point
                quantized_image_batch = np.round(quantized_image_batch)
                if input_dtype == np.int8: quantized_image_batch = np.clip(quantized_image_batch, -128, 127)
                elif input_dtype == np.uint8: quantized_image_batch = np.clip(quantized_image_batch, 0, 255)
                processed_input = quantized_image_batch.astype(input_dtype)
            else:
                processed_input = float_image_batch.astype(input_dtype)

            final_input = processed_input
            expected_shape = tuple(input_details[0]['shape'])
            current_shape = processed_input.shape
            if len(expected_shape) == 4 and expected_shape[3] == 1 and len(current_shape) == 3: # e.g. (1, 28, 28, 1) vs (1, 28, 28)
                 final_input = np.expand_dims(final_input, axis=-1)
            elif len(expected_shape) == 2 and expected_shape[0] == 1 and len(current_shape) > 1 and expected_shape[1] == np.prod(current_shape[1:]): # e.g. (1, 784) vs (1, 28, 28)
                final_input = final_input.reshape(1, -1)

            tflite_interpreter.set_tensor(input_details[0]['index'], final_input)
            tflite_interpreter.invoke()
            output_data = tflite_interpreter.get_tensor(output_details[0]['index'])

            if is_output_quantized:
                output_data_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
            else:
                output_data_float = output_data
            if np.argmax(output_data_float) == y_test_data[i]:
                num_correct += 1
        accuracy = num_correct / num_total
        print(f"Test Accuracy: {accuracy:.4f} (Note: Loss for TFLite models is not directly calculated here)")
        return 0.0, accuracy 
    else: 
        loss, accuracy = model.evaluate(x_test_data, y_test_data, verbose=0)
        print(f"Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")
        return loss, accuracy

def print_weight_statistics(model, name):
    print(f"\nWeight Statistics for {name}:")
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            weights_list = layer.get_weights()
            if weights_list: # Check if weights exist
                weights = weights_list[0]
                print(f" Layer {i} ({layer.name}): Min={np.min(weights):.4f}, Max={np.max(weights):.4f}, Mean={np.mean(weights):.4f}, Std={np.std(weights):.4f}")
            else:
                print(f" Layer {i} ({layer.name}): No weights found.")
        elif isinstance(layer, tfmot.quantization.keras.QuantizeWrapperV2):
            try:
                # Attempt to get the underlying float weights (these are what TF MOT tunes)
                unquantized_weights = None
                for weight_var in layer.trainable_weights: # 학습 가능한 가중치 중에서 kernel을 찾음
                    if 'kernel' in weight_var.name and 'quantize_layer' not in weight_var.name: # Avoid quantizer variables
                        unquantized_weights = weight_var.numpy()
                        break
                if unquantized_weights is not None:
                    print(f" Layer {i} ({layer.name} - QAT Float Kernel): Min={np.min(unquantized_weights):.4f}, Max={np.max(unquantized_weights):.4f}, Mean={np.mean(unquantized_weights):.4f}, Std={np.std(unquantized_weights):.4f}")
                else:
                    print(f" Layer {i} ({layer.name} - QAT): Could not extract comparable float kernel weight stats.")
            except Exception as e:
                 print(f" Layer {i} ({layer.name} - QAT): Error extracting weight stats: {e}")

def plot_metric_histogram(data, title, xlabel, ax=None, bins=50):
    if ax is None:
        fig_standalone, current_ax = plt.subplots(figsize=(8, 5))
    else:
        current_ax = ax
    
    if not data: # Check if data is None or empty
        current_ax.text(0.5, 0.5, "No data for histogram", ha='center', va='center', transform=current_ax.transAxes)
    else:
        data_array = np.array(data)
        if data_array.size == 0 : # Check if array is empty after conversion
            current_ax.text(0.5, 0.5, "No data for histogram (empty array)", ha='center', va='center', transform=current_ax.transAxes)
        else:
            current_ax.hist(data_array, bins=bins, edgecolor='black')
    
    current_ax.set_title(title, fontsize=10)
    current_ax.set_xlabel(xlabel, fontsize=8)
    current_ax.set_ylabel("Frequency", fontsize=8)
    current_ax.tick_params(axis='both', which='major', labelsize=7)
    current_ax.grid(axis='y', alpha=0.75)
    
    if ax is None:
        plt.show(block=False)
        plt.pause(0.1)


def plot_weight_hist(model, title="Weight Distribution", bins=50, ax=None, 
                     tflite_interpreter=None, plot_dequantized=False):
    current_ax = ax
    if current_ax is None:
        fig_standalone, current_ax = plt.subplots(figsize=(8, 5))

    w_all = []
    x_label = "Weight Value"
    is_tflite_model = model is None and tflite_interpreter is not None

    if is_tflite_model:
        tensor_dtype_str = "unknown"
        first_output_idx = tflite_interpreter.get_output_details()[0]['index'] # Get first output index once
        for i in range(len(tflite_interpreter.get_tensor_details())):
            tensor_details = tflite_interpreter.get_tensor_details()[i]
            
            is_weight_tensor = ('kernel' in tensor_details['name'] or 'weights' in tensor_details['name']) and \
                               ('bias' not in tensor_details['name']) and \
                               (tensor_details['index'] < first_output_idx) # Compare with the stored first_output_idx
            
            if is_weight_tensor and tensor_details['shape'].size > 1 : 
                tensor_data = tflite_interpreter.get_tensor(tensor_details['index'])
                tensor_dtype_str = str(tensor_data.dtype)
                if tensor_data.ndim > 0:
                    if plot_dequantized and tensor_details.get('quantization_parameters') and \
                       tensor_details['quantization_parameters']['scales'] and \
                       len(tensor_details['quantization_parameters']['scales']) > 0 and \
                       tensor_details['quantization_parameters']['zero_points'] and \
                       len(tensor_details['quantization_parameters']['zero_points']) > 0:
                        
                        scale = tensor_details['quantization_parameters']['scales'][0]
                        zero_point = tensor_details['quantization_parameters']['zero_points'][0]
                        
                        if scale != 0:
                           dequantized_data = (tensor_data.astype(np.float32) - zero_point) * scale
                           w_all.extend(dequantized_data.flatten())
                        else: 
                           w_all.extend(tensor_data.flatten())
                    else: 
                        w_all.extend(tensor_data.flatten())
        x_label = f"Weight Value ({'Dequantized F32' if plot_dequantized and w_all else 'Raw ' + tensor_dtype_str})"


    elif model is not None: 
        for layer in model.layers:
            layer_kernel_weights = None
            if isinstance(layer, tfmot.quantization.keras.QuantizeWrapperV2):
                for weight_variable in layer.weights: # Use layer.weights to get all, then filter
                    if "kernel" in weight_variable.name and "quantize_layer" not in weight_variable.name and "bias" not in weight_variable.name:
                        # Check if it's the actual float kernel that QAT tunes
                        # The actual float weights are usually an attribute of the wrapper or its inner layer.
                        # For TF MOT QAT, the underlying float weights are what get trained.
                        # Let's try to get the weights of the wrapped layer if possible or look for float kernel directly
                        try:
                            # This assumes the float kernel is among trainable_weights and named appropriately
                            float_kernel_candidates = [w for w in layer.trainable_weights if 'kernel' in w.name and 'quantizer' not in w.name.lower()]
                            if float_kernel_candidates:
                                layer_kernel_weights = float_kernel_candidates[0].numpy().flatten()
                            else: # Fallback if specific float kernel not found among trainable_weights
                                if hasattr(layer.layer, 'kernel'): # Access kernel of the wrapped layer
                                     layer_kernel_weights = layer.layer.kernel.numpy().flatten()
                        except AttributeError:
                            pass # Could not get weights this way
                        if layer_kernel_weights is not None: break 
            elif isinstance(layer, tf.keras.layers.Dense):
                if layer.get_weights():
                    layer_kernel_weights = layer.get_weights()[0].flatten()
            
            if layer_kernel_weights is not None:
                w_all.extend(layer_kernel_weights)
        x_label = "Weight Value (Float32 from Keras)"
    
    if not w_all:
        current_ax.text(0.5, 0.5, "No weights found/extracted", ha='center', va='center', transform=current_ax.transAxes)
    else:
        current_ax.hist(np.array(w_all), bins=bins, edgecolor='black')
        
    current_ax.set_title(title, fontsize=10)
    current_ax.set_xlabel(x_label, fontsize=8)
    current_ax.set_ylabel("Count", fontsize=8)
    current_ax.tick_params(axis='both', which='major', labelsize=7)
    current_ax.grid(axis='y', alpha=0.75)

    if ax is None:
        plt.show(block=False)
        plt.pause(0.1)
#%%
# --- Main Experiment ---
results = {}
model_history_callbacks = {} 

TOTAL_MAIN_PLOTS = 10 
NUM_ROWS = 4 
NUM_COLS = 3 
fig_main, all_axes_main = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(14, 12)) 
axes_flat_main = all_axes_main.flatten()
plot_idx_main = 0

conductance_handler = ConductanceManager(
    filepath=config["filepath"], mode=config["mode"], clip=config["clip"],
    use_scaling=config["use_scaling"], scaling_factor=config["scaling_factor"],
    noise_scaling_factor=config["noise_scaling_factor"],
    use_int_levels=config["use_int_levels"], int_min=config["int_min"], int_max=config["int_max"],
    num_of_G=config.get("num_of_G", -1), G_sampling_method=config.get("G_sampling_method", "uniform_interval"),
    apply_on_batch=config.get("apply_on_batch", False) # Pass apply_on_batch here if it's meant for the manager too
)

def record_initial_perf(model, model_name_suffix, x_data, y_data, results_dict):
    loss, acc = model.evaluate(x_data, y_data, verbose=0)
    key = f"Initial_{model_name_suffix}"
    results_dict[key] = {"loss": loss, "accuracy": acc}
    print(f"{key}: Loss={loss:.4f}, Accuracy={acc:.4f}")

def plot_metric_history(history_callback, metric_plot_configs, ax, title_prefix=""):
    epochs_ran = 0
    if history_callback and history_callback.epoch_train_losses:
        epochs_ran = len(history_callback.epoch_train_losses)
    
    if epochs_ran == 0:
        ax.text(0.5, 0.5, "No training data for history plot", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title_prefix}History (No Data)", fontsize=9)
        return

    epoch_range = range(1, epochs_ran + 1)
    
    combined_labels = []
    y_label_parts = []

    for data_key, label, color in metric_plot_configs:
        metric_data = getattr(history_callback, data_key, [])
        if metric_data and len(metric_data) == epochs_ran:
            ax.plot(epoch_range, metric_data, marker='.', linestyle='-', label=label, color=color)
            combined_labels.append(label)
            if "Acc" in label and "Accuracy" not in y_label_parts:
                y_label_parts.append("Accuracy")
            if "Loss" in label and "Loss" not in y_label_parts:
                y_label_parts.append("Loss")
        else:
            print(f"[plot_metric_history] Data missing or mismatched for {data_key} (expected {epochs_ran}, got {len(metric_data)})")


    if combined_labels:
        ax.set_title(f"{title_prefix}{'/'.join(combined_labels)} vs Epochs", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(' / '.join(y_label_parts) if y_label_parts else "Metric Value", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.6)
    else:
        ax.text(0.5, 0.5, "No data for selected metrics", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title_prefix}History (Metrics Missing)", fontsize=9)


#%%
# --- Experiment Execution (Order Changed) ---

# 1. Baseline Float32 Model
print("\n=== 1. Baseline Float32 Model ===")
model_baseline = create_mnist_model(kernel_constraint=conductance_handler.get_kernel_constraint())
model_baseline.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
record_initial_perf(model_baseline, "Baseline_Float32", x_test, y_test, results)
baseline_callbacks = [BatchMetricsCallback(print_batch_loss_freq=0)]
model_baseline.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"],
                   validation_data=(x_test,y_test), callbacks=baseline_callbacks, verbose=0)
loss, acc = evaluate_model(model_baseline, "Baseline Float32 (Trained)", x_test, y_test)
results["Baseline_Float32_Trained"] = {"loss": loss, "accuracy": acc}
model_history_callbacks["Baseline_Float32"] = baseline_callbacks[0]
print_weight_statistics(model_baseline, "Baseline Float32 (Trained)")
if plot_idx_main < len(axes_flat_main):
    plot_weight_hist(model_baseline, "1. Baseline Float32 Weights", bins=100, ax=axes_flat_main[plot_idx_main])
    plot_idx_main += 1

# 2. Custom PTQ (G_mean mapping Only)
print("\n=== 2. Custom PTQ (G_mean mapping Only) ===")
model_ptq_g_mean_only = tf.keras.models.clone_model(model_baseline)
model_ptq_g_mean_only.set_weights(model_baseline.get_weights())
model_ptq_g_mean_only.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
record_initial_perf(model_ptq_g_mean_only, "Custom_PTQ_G_mean_Only_Before_Map", x_test, y_test, results)
print("Mapping weights to G_mean (NO noise)...")
conductance_handler.map_weights_to_g_mean(model_ptq_g_mean_only)
loss_ptq_gmo, acc_ptq_gmo = evaluate_model(model_ptq_g_mean_only, "Custom PTQ (G_mean Only)", x_test, y_test)
results["Custom_PTQ_G_mean_Only_Mapped"] = {"loss": loss_ptq_gmo, "accuracy": acc_ptq_gmo}
print_weight_statistics(model_ptq_g_mean_only, "Custom PTQ (G_mean Only)")
conductance_handler.check_weights_mapped(model_ptq_g_mean_only, tolerance=1e-7)
if plot_idx_main < len(axes_flat_main):
    plot_weight_hist(model_ptq_g_mean_only, f"2. Custom PTQ G_mean Only", bins=100, ax=axes_flat_main[plot_idx_main])
    plot_idx_main += 1

# 3. Framework QAT (TF MOT for int8) - Keras model part
print("\n=== 3. Framework QAT (TF MOT Keras Model) ===")
model_qat_framework_base = create_mnist_model()
q_aware_model = quantize_model(model_qat_framework_base) 
q_aware_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
record_initial_perf(q_aware_model, "Framework_QAT_TF MOT_Keras", x_test, y_test, results)
framework_qat_hist_cb = BatchMetricsCallback(print_batch_loss_freq=0)
q_aware_model.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_data=(x_test, y_test),
                  callbacks=[framework_qat_hist_cb], verbose=0)
loss_qat_fw, acc_qat_fw = evaluate_model(q_aware_model, "Framework QAT (TF MOT, Keras Trained)", x_test, y_test)
results["Framework_QAT_TF MOT_Keras_Trained"] = {"loss": loss_qat_fw, "accuracy": acc_qat_fw}
model_history_callbacks["Framework_QAT_TF MOT_Keras"] = framework_qat_hist_cb
print_weight_statistics(q_aware_model, "Framework QAT (TF MOT Keras Trained)")
if plot_idx_main < len(axes_flat_main):
    plot_weight_hist(q_aware_model, "3. Framework QAT (Keras) Weights", bins=100, ax=axes_flat_main[plot_idx_main])
    plot_idx_main += 1

# 4. Framework QAT (TF MOT TFLite INT8 part)
print("\n=== 4. Framework QAT (TF MOT TFLite INT8 Conversion) ===")
if 'Framework_QAT_TF MOT_Keras_Trained' in results: 
    try:
        converter_qat = tf.lite.TFLiteConverter.from_keras_model(q_aware_model) 
        converter_qat.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_qat.representative_dataset = representative_data_gen
        converter_qat.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_qat.inference_input_type = tf.int8
        converter_qat.inference_output_type = tf.int8
        tflite_model_qat = converter_qat.convert()
        with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmpfile:
            tmpfile.write(tflite_model_qat)
            tflite_model_qat_path = tmpfile.name
        interpreter_qat_tflite = tf.lite.Interpreter(model_path=tflite_model_qat_path)
        interpreter_qat_tflite.allocate_tensors()
        loss_qat_tfl, acc_qat_tfl = evaluate_model(None, "Framework QAT (TFLite INT8)", x_test, y_test, is_tflite=True, tflite_interpreter=interpreter_qat_tflite)
        results["Framework_QAT_TFLite_INT8"] = {"loss": loss_qat_tfl, "accuracy": acc_qat_tfl}
        if plot_idx_main < len(axes_flat_main):
            plot_weight_hist(None, "4. Framework QAT TFLite INT8 Weights", bins=256,
                             ax=axes_flat_main[plot_idx_main], tflite_interpreter=interpreter_qat_tflite, plot_dequantized=True)
            plot_idx_main += 1
        os.remove(tflite_model_qat_path)
    except Exception as e:
        print(f"Framework QAT (TFLite INT8 conversion/evaluation) failed: {e}")
        results["Framework_QAT_TFLite_INT8"] = {"loss": -1, "accuracy": -1, "error": str(e)}
        if plot_idx_main < len(axes_flat_main):
            axes_flat_main[plot_idx_main].text(0.5, 0.5, f"TFLite INT8 Failed", ha='center', va='center', fontsize=8, color='red')
            axes_flat_main[plot_idx_main].set_title("4. Framework QAT TFLite (Failed)", fontsize=10)
            plot_idx_main += 1
else:
    print("Skipping Framework QAT TFLite conversion as Keras QAT model training might have failed or was skipped.")
    if plot_idx_main < len(axes_flat_main):
        axes_flat_main[plot_idx_main].text(0.5, 0.5, "Skipped: TFLite Conversion", ha='center', va='center', fontsize=9)
        axes_flat_main[plot_idx_main].set_title("4. Framework QAT TFLite (Skipped)", fontsize=10)
        plot_idx_main += 1

# 5. Custom QAT (G_mean mapping Only, noise_scaling_factor=0 for this QAT part)
print("\n=== 5. Custom QAT (G_mean mapping Only during training) ===")
original_noise_factor = conductance_handler.noise_scaling_factor # Save original
conductance_handler.noise_scaling_factor = 0 # Set to 0 for this experiment
model_qat_g_mean_only = create_mnist_model(kernel_constraint=conductance_handler.get_kernel_constraint())
model_qat_g_mean_only.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
record_initial_perf(model_qat_g_mean_only, "Custom_QAT_G_mean_Only", x_test, y_test, results)
# Pass the global 'config' dictionary here
custom_qat_g_mean_cb = conductance_handler.get_custom_qat_callback(model_qat_g_mean_only, config, inject_noise_during_training=False)
qat_g_mean_hist_cb = BatchMetricsCallback(print_batch_loss_freq=0)
model_qat_g_mean_only.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_data=(x_test, y_test),
                          callbacks=[custom_qat_g_mean_cb, qat_g_mean_hist_cb], verbose=0)
loss, acc = evaluate_model(model_qat_g_mean_only, "Custom QAT (G_mean Only Trained)", x_test, y_test)
results["Custom_QAT_G_mean_Only_Trained"] = {"loss": loss, "accuracy": acc}
model_history_callbacks["Custom_QAT_G_mean_Only"] = qat_g_mean_hist_cb
print_weight_statistics(model_qat_g_mean_only, "Custom QAT (G_mean Only Trained)")
conductance_handler.check_weights_mapped(model_qat_g_mean_only, tolerance=1e-5)
if plot_idx_main < len(axes_flat_main):
    plot_weight_hist(model_qat_g_mean_only, f"5. Custom QAT G_mean Only", bins=100, ax=axes_flat_main[plot_idx_main])
    plot_idx_main += 1
conductance_handler.noise_scaling_factor = original_noise_factor # Restore original

# 6. Custom QAT (G_mean mapping + Noise during training)
print("\n=== 6. Custom QAT (G_mean mapping + Noise during training) ===")
# Ensure noise_scaling_factor in the global config is used for this decision
if config["noise_scaling_factor"] > 0:
    # It's important that conductance_handler.noise_scaling_factor reflects the desired state for this experiment.
    # If it was changed above and not restored, or if it should be different from global config for this specific run, adjust here.
    # Assuming conductance_handler.noise_scaling_factor is already correctly set (e.g., to global config["noise_scaling_factor"])
    # or restore it if it was changed by a previous experiment and not reset.
    # conductance_handler.noise_scaling_factor = config["noise_scaling_factor"] # Explicitly set if unsure

    model_qat_custom_noise_train = create_mnist_model(kernel_constraint=conductance_handler.get_kernel_constraint())
    model_qat_custom_noise_train.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    record_initial_perf(model_qat_custom_noise_train, "Custom_QAT_G_mean_Noise", x_test, y_test, results)
    # Pass the global 'config' dictionary here
    custom_qat_noise_cb = conductance_handler.get_custom_qat_callback(model_qat_custom_noise_train, config, inject_noise_during_training=True)
    qat_noise_hist_cb = BatchMetricsCallback(print_batch_loss_freq=0)
    model_qat_custom_noise_train.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_data=(x_test, y_test),
                               callbacks=[custom_qat_noise_cb, qat_noise_hist_cb], verbose=0)
    loss, acc = evaluate_model(model_qat_custom_noise_train, "Custom QAT (G_mean + Noise Trained)", x_test, y_test)
    results["Custom_QAT_G_mean_Noise_Trained"] = {"loss": loss, "accuracy": acc}
    model_history_callbacks["Custom_QAT_G_mean_Noise"] = qat_noise_hist_cb
    print_weight_statistics(model_qat_custom_noise_train, "Custom QAT (G_mean + Noise Trained)")
    # Tolerance for check_weights_mapped might need to be larger if noise is aggressive
    conductance_handler.check_weights_mapped(model_qat_custom_noise_train, tolerance=conductance_handler.noise_scaling_factor * 1e-1 if conductance_handler.noise_scaling_factor > 0 else 1e-5)
    if plot_idx_main < len(axes_flat_main):
        plot_weight_hist(model_qat_custom_noise_train, f"6. Custom QAT G_mean + Noise", bins=100, ax=axes_flat_main[plot_idx_main])
        plot_idx_main += 1
else:
    print(f"Skipping Custom QAT with Noise as config[\"noise_scaling_factor\"] is {config['noise_scaling_factor']}.")
    if plot_idx_main < len(axes_flat_main):
        axes_flat_main[plot_idx_main].text(0.5, 0.5, "Skipped: Custom QAT with Noise", ha='center', va='center', fontsize=9)
        axes_flat_main[plot_idx_main].set_title("6. Custom QAT G_mean + Noise (Skipped)", fontsize=10)
        plot_idx_main += 1

# 7. Custom PTQ (G_mean mapping + Noise)
print("\n=== 7. Custom PTQ (G_mean mapping + Noise) ===")
# Ensure conductance_handler.noise_scaling_factor is set from global config for this PTQ step
# This is important if it was modified by a QAT step.
conductance_handler.noise_scaling_factor = config["noise_scaling_factor"]

model_ptq_custom_noise = tf.keras.models.clone_model(model_baseline) 
model_ptq_custom_noise.set_weights(model_baseline.get_weights())
model_ptq_custom_noise.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Mapping weights to G_mean...")
conductance_handler.map_weights_to_g_mean(model_ptq_custom_noise)
if conductance_handler.noise_scaling_factor > 0: # Check the handler's current state
    print("Adding noise based on G_std...")
    conductance_handler.add_noise_to_weights(model_ptq_custom_noise)
else:
    print("Noise scaling factor is 0, skipping noise addition for PTQ.")
loss, acc = evaluate_model(model_ptq_custom_noise, "Custom PTQ (G_mean + Noise)", x_test, y_test)
results["Custom_PTQ_G_mean_Noise_Mapped"] = {"loss": loss, "accuracy": acc} 
print_weight_statistics(model_ptq_custom_noise, "Custom PTQ (G_mean + Noise)")
conductance_handler.check_weights_mapped(model_ptq_custom_noise, tolerance=conductance_handler.noise_scaling_factor * 1e-2 if conductance_handler.noise_scaling_factor > 0 else 1e-7)
if plot_idx_main < len(axes_flat_main):
    plot_weight_hist(model_ptq_custom_noise, f"7. Custom PTQ G_mean + Noise", bins=100, ax=axes_flat_main[plot_idx_main])
    plot_idx_main += 1


# Add training history plots for Baseline model
baseline_cb_data = model_history_callbacks.get("Baseline_Float32")
if baseline_cb_data:
    if plot_idx_main < len(axes_flat_main):
        plot_metric_history(baseline_cb_data,
                            [('epoch_train_accuracies', 'Train Acc', 'tab:blue'), 
                             ('epoch_val_accuracies', 'Val Acc', 'tab:orange')],
                            ax=axes_flat_main[plot_idx_main], title_prefix="Baseline: ")
        plot_idx_main +=1
    else: print("Not enough subplot space for Baseline Acc History")

    if plot_idx_main < len(axes_flat_main):
         plot_metric_history(baseline_cb_data,
                            [('epoch_train_losses', 'Train Loss', 'tab:green'), 
                             ('epoch_val_losses', 'Val Loss', 'tab:red')],
                            ax=axes_flat_main[plot_idx_main], title_prefix="Baseline: ")
         plot_idx_main +=1
    else: print("Not enough subplot space for Baseline Loss History")

    if plot_idx_main < len(axes_flat_main) and baseline_cb_data.batch_losses:
        plot_metric_histogram(baseline_cb_data.batch_losses, "Baseline: Batch Losses Dist.", "Batch Loss Value", ax=axes_flat_main[plot_idx_main], bins=50)
        plot_idx_main += 1
    else: 
        if plot_idx_main < len(axes_flat_main): 
            axes_flat_main[plot_idx_main].text(0.5, 0.5, "Batch Loss Hist Skipped", ha='center', va='center', fontsize=9)
            axes_flat_main[plot_idx_main].set_title("Batch Loss Hist (Skipped)", fontsize=10)
            plot_idx_main += 1
        else:
            print("Not enough subplot space for Batch Loss Histogram or no data")
else:
    print("Baseline callback data not found for history plots.")
    for _ in range(3): 
        if plot_idx_main < len(axes_flat_main):
            axes_flat_main[plot_idx_main].text(0.5, 0.5, "Plot Skipped\n(No History Data)", ha='center', va='center', fontsize=9)
            axes_flat_main[plot_idx_main].set_title("History Plot (Skipped)", fontsize=10)
            plot_idx_main += 1


# --- Clean up unused subplots in the main figure ---
for i in range(plot_idx_main, NUM_ROWS * NUM_COLS):
    if i < len(axes_flat_main): # Ensure index is valid before deleting
      fig_main.delaxes(axes_flat_main[i])
fig_main.suptitle(f"MNIST Quantization Experiments (G_levels={config['num_of_G']}, NoiseFactor={config['noise_scaling_factor']})", fontsize=16)
fig_main.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect to prevent suptitle overlap


# --- Experiment: num_of_G variation ---
print("\n\n=== Experiment: Impact of Number of Conductance Levels (num_of_G) ===")
print("Target Model for this experiment: Custom PTQ (G_mean mapping only)")
# Ensure config.get("num_of_G", 16) is part of the set for comparison
default_num_g_from_config = config.get("num_of_G", 16)
if default_num_g_from_config == -1: default_num_g_from_config = 100
num_of_G_values = sorted(list(set([1, 2, 4, 8, 16, 32, 64, 100, default_num_g_from_config])), reverse=True)

num_g_results = {"num_g": [], "accuracy": [], "loss": []}

epochs_for_num_g_exp = 3 # Using a smaller number of epochs for this sweep
print(f"Training a baseline model for {epochs_for_num_g_exp} epochs to be used for PTQ in num_G sweep...")
model_baseline_for_num_g_exp = create_mnist_model()
model_baseline_for_num_g_exp.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model_baseline_for_num_g_exp.fit(x_train, y_train, epochs=epochs_for_num_g_exp, batch_size=config["batch_size"], validation_split=0.1, verbose=1)
print("Baseline training for num_G sweep complete.")

for num_g_val in num_of_G_values:
    print(f"\n--- Testing with num_of_G = {num_g_val} ---")
    
    temp_conductance_handler = ConductanceManager(
        filepath=config["filepath"], mode=config["mode"], clip=config["clip"],
        use_scaling=config["use_scaling"], scaling_factor=config["scaling_factor"],
        noise_scaling_factor=0, # PTQ for this experiment, so no noise factor during G-level setup
        use_int_levels=config["use_int_levels"], int_min=config["int_min"], int_max=config["int_max"],
        num_of_G=num_g_val, G_sampling_method=config["G_sampling_method"]
    )
    if temp_conductance_handler.G_mean is None or not temp_conductance_handler.G_mean.size:
        print(f"Skipping num_of_G = {num_g_val} due to empty G_mean levels from ConductanceManager.")
        num_g_results["num_g"].append(num_g_val) # Record that we tried
        num_g_results["accuracy"].append(np.nan) # Indicate failure/skip
        num_g_results["loss"].append(np.nan)
        continue

    model_ptq_num_g = tf.keras.models.clone_model(model_baseline_for_num_g_exp)
    model_ptq_num_g.set_weights(model_baseline_for_num_g_exp.get_weights())
    model_ptq_num_g.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    temp_conductance_handler.map_weights_to_g_mean(model_ptq_num_g)
    
    loss, acc = evaluate_model(model_ptq_num_g, f"PTQ (G_mean Only) num_G={num_g_val}", x_test, y_test)
    num_g_results["num_g"].append(num_g_val)
    num_g_results["accuracy"].append(acc)
    num_g_results["loss"].append(loss if loss is not None else np.nan)


if num_g_results["num_g"] and np.sum(~np.isnan(num_g_results["accuracy"])) > 0 : # Check if any valid accuracy data
    fig_num_g, ax_acc_num_g = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax_acc_num_g.set_xlabel('Number of Conductance Levels (num_of_G)')
    ax_acc_num_g.set_ylabel('Accuracy', color=color)
    ax_acc_num_g.plot(num_g_results["num_g"], num_g_results["accuracy"], color=color, marker='o', linestyle='-', label='Accuracy')
    ax_acc_num_g.tick_params(axis='y', labelcolor=color)
    
    # Use log scale if range of num_g is large
    if len(num_g_results["num_g"]) > 1 and max(num_g_results["num_g"]) / min(num_g_results["num_g"]) > 10 :
         ax_acc_num_g.set_xscale('log')
    ax_acc_num_g.invert_xaxis() # Typically higher num_g is better, so show from high to low or adjust as preferred

    ax_loss_num_g = ax_acc_num_g.twinx()
    color = 'tab:red'
    ax_loss_num_g.set_ylabel('Loss', color=color)
    # valid_losses = [l if isinstance(l, (int,float)) and l != 0.0 else np.nan for l in num_g_results["loss"]] 
    ax_loss_num_g.plot(num_g_results["num_g"], num_g_results["loss"], color=color, marker='x', linestyle='--', label='Loss')
    ax_loss_num_g.tick_params(axis='y', labelcolor=color)

    fig_num_g.suptitle('Impact of num_of_G on Custom PTQ (G-mean only)', fontsize=12)
    lines, labels = ax_acc_num_g.get_legend_handles_labels()
    lines2, labels2 = ax_loss_num_g.get_legend_handles_labels()
    ax_loss_num_g.legend(lines + lines2, labels + labels2, loc='best')
    
    ax_acc_num_g.grid(True, which="both", ls="-", alpha=0.5)
    fig_num_g.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)
else:
    print("Not enough data to plot num_of_G impact.")
    fig_num_g = None # Ensure fig_num_g exists for save check


# --- Final Results Summary ---
print("\n\n--- Experiment Summary (Main Set) ---")
if config["use_int_levels"]:
    print(f"Mode: Integer Levels ({config['int_min']} to {config['int_max']})")
else:
    print(f"Mode: CSV Conductance Data (Path: {config['filepath']})")
print(f"G_mean Levels (from config): {config['num_of_G']}, Sampling: {config['G_sampling_method']}")
print(f"Scaling: {config['use_scaling']}, Factor: {config['scaling_factor']}")
print(f"Noise Scaling Factor for Custom Models: {config['noise_scaling_factor']}")
print(f"Epochs (main experiments): {config['epochs']}")

sorted_results_keys = sorted(results.keys())
for model_name in sorted_results_keys:
    metrics = results[model_name]
    if "error" in metrics: # Check if error key exists
        print(f"{model_name}: Accuracy = {metrics.get('accuracy', 'N/A')}, Loss = {metrics.get('loss', 'N/A')} (Error: {metrics['error']})")
    else:
        loss_value = metrics.get('loss')
        loss_str = f"{loss_value:.4f}" if isinstance(loss_value, (int, float)) and np.isfinite(loss_value) else "N/A"
        
        acc_value = metrics.get('accuracy')
        acc_str = f"{acc_value:.4f}" if isinstance(acc_value, (int, float)) and np.isfinite(acc_value) else "N/A"
        print(f"{model_name}: Accuracy = {acc_str}, Loss = {loss_str}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_save_dir = os.path.expanduser("~/python/quantization/results/") # Adjusted path for common use
os.makedirs(base_save_dir, exist_ok=True)
experiment_name_parts = [
    "mnist",
    f"mode{config['mode']}",
    f"g{config['num_of_G']}",
    f"noise{config['noise_scaling_factor']}",
    timestamp
]
save_dir_name = "_".join(experiment_name_parts)
save_dir = os.path.join(base_save_dir, save_dir_name)
os.makedirs(save_dir, exist_ok=True)

config_save_path = os.path.join(save_dir, "config.json")
results_save_path = os.path.join(save_dir, "results.json")
main_plot_save_path = os.path.join(save_dir, "main_plots.png")
num_g_plot_save_path = os.path.join(save_dir, "num_g_impact_plot.png")

with open(config_save_path, "w") as f:
    serializable_config = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in config.items()}
    json.dump(serializable_config, f, indent=4)

with open(results_save_path, "w") as f:
    serializable_results = {}
    for k, v_dict in results.items():
        serializable_results[k] = {
            sub_k: (
                float(sub_v) if isinstance(sub_v, (np.floating, float)) and np.isfinite(sub_v) 
                else str(sub_v) # Store non-finite floats (NaN, inf) as strings or handle as preferred
            ) for sub_k, sub_v in v_dict.items()
        }
    json.dump(serializable_results, f, indent=4)

print(f"\nConfiguration and results saved to: {save_dir}")
if 'fig_main' in locals() and fig_main is not None: # Check if fig_main was created
    try:
        fig_main.savefig(main_plot_save_path)
        print(f"Main experiment plots saved to: {main_plot_save_path}")
    except Exception as e:
        print(f"Error saving main plot: {e}")

if 'fig_num_g' in locals() and fig_num_g is not None: # Check if fig_num_g was created
    try:
        fig_num_g.savefig(num_g_plot_save_path)
        print(f"Num_of_G impact plot saved to: {num_g_plot_save_path}")
    except Exception as e:
        print(f"Error saving num_g plot: {e}")

# --- Experiment: Impact of Number of Conductance Levels (num_of_G) with NOISE ---
print("\n\n=== Experiment: Impact of Number of Conductance Levels (num_of_G) with NOISE ===")
print("Target Model for this experiment: Custom PTQ (G-mean mapping + Noise)")
# num_of_G_values 와 model_baseline_for_num_g_exp 는 이전 실험에서 정의된 것을 재사용합니다.

num_g_noise_results = {"num_g": [], "accuracy": [], "loss": []}

# config에서 noise_scaling_factor 가져오기 (0보다 클 때만 노이즈 적용)
noise_factor_for_sweep = config.get("noise_scaling_factor", 0)

if noise_factor_for_sweep > 0:
    for num_g_val in num_of_G_values: # 이전 실험과 동일한 num_of_G 값들 사용
        print(f"\n--- Testing with num_of_G = {num_g_val} (with Noise) ---")
        
        # 이 실험용 ConductanceManager 생성 시 noise_scaling_factor를 config 값으로 설정
        temp_conductance_handler_noise = ConductanceManager(
            filepath=config["filepath"], 
            mode=config["mode"], 
            clip=config["clip"], # clip 설정은 config 따름
            use_scaling=config["use_scaling"], 
            scaling_factor=config["scaling_factor"],
            noise_scaling_factor=noise_factor_for_sweep, # ★★★ 설정된 노이즈 강도 사용 ★★★
            use_int_levels=config["use_int_levels"], 
            int_min=config["int_min"], 
            int_max=config["int_max"],
            num_of_G=num_g_val, 
            G_sampling_method=config["G_sampling_method"]
        )
        
        if temp_conductance_handler_noise.G_mean is None or not temp_conductance_handler_noise.G_mean.size:
            print(f"Skipping num_of_G = {num_g_val} (with Noise) due to empty G_mean levels.")
            num_g_noise_results["num_g"].append(num_g_val)
            num_g_noise_results["accuracy"].append(np.nan)
            num_g_noise_results["loss"].append(np.nan)
            continue

        model_ptq_num_g_noise = tf.keras.models.clone_model(model_baseline_for_num_g_exp)
        model_ptq_num_g_noise.set_weights(model_baseline_for_num_g_exp.get_weights())
        model_ptq_num_g_noise.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
        print(f"  Mapping weights to G_mean for num_G={num_g_val} (with Noise)...")
        temp_conductance_handler_noise.map_weights_to_g_mean(model_ptq_num_g_noise)
        
        print(f"  Adding noise for num_G={num_g_val}...")
        temp_conductance_handler_noise.add_noise_to_weights(model_ptq_num_g_noise) # ★★★ 노이즈 추가 ★★★
        
        loss, acc = evaluate_model(model_ptq_num_g_noise, f"PTQ (G_mean + Noise) num_G={num_g_val}", x_test, y_test)
        num_g_noise_results["num_g"].append(num_g_val)
        num_g_noise_results["accuracy"].append(acc)
        num_g_noise_results["loss"].append(loss if loss is not None else np.nan)

    # Plotting for "Impact of num_of_G on Custom PTQ (G-mean + Noise)"
    if num_g_noise_results["num_g"] and np.sum(~np.isnan(num_g_noise_results["accuracy"])) > 0 :
        fig_num_g_noise, ax_acc_num_g_noise = plt.subplots(figsize=(10, 6))
        color = 'tab:blue'
        ax_acc_num_g_noise.set_xlabel('Number of Conductance Levels (num_of_G)')
        ax_acc_num_g_noise.set_ylabel('Accuracy', color=color)
        ax_acc_num_g_noise.plot(num_g_noise_results["num_g"], num_g_noise_results["accuracy"], color=color, marker='o', linestyle='-', label='Accuracy')
        ax_acc_num_g_noise.tick_params(axis='y', labelcolor=color)
        
        if len(num_g_noise_results["num_g"]) > 1: # x_scale log 조건은 기존과 동일하게 적용 가능
            # min_val = min(m for m in num_g_noise_results["num_g"] if m > 0) # 0보다 큰 최소값으로 나눠야 함
            # if max(num_g_noise_results["num_g"]) / min_val > 10:
            valid_num_g = [n for n in num_g_noise_results["num_g"] if n > 0]
            if valid_num_g and max(valid_num_g) / min(valid_num_g) > 10:
                 ax_acc_num_g_noise.set_xscale('log')
        ax_acc_num_g_noise.invert_xaxis() 

        ax_loss_num_g_noise = ax_acc_num_g_noise.twinx()
        color = 'tab:red'
        ax_loss_num_g_noise.set_ylabel('Loss', color=color)
        ax_loss_num_g_noise.plot(num_g_noise_results["num_g"], num_g_noise_results["loss"], color=color, marker='x', linestyle='--', label='Loss')
        ax_loss_num_g_noise.tick_params(axis='y', labelcolor=color)

        fig_num_g_noise.suptitle('Impact of num_of_G on Custom PTQ (G-mean + Noise)', fontsize=12) # ★★★ 제목 변경 ★★★
        lines, labels = ax_acc_num_g_noise.get_legend_handles_labels()
        lines2, labels2 = ax_loss_num_g_noise.get_legend_handles_labels()
        ax_loss_num_g_noise.legend(lines + lines2, labels + labels2, loc='best')
        
        ax_acc_num_g_noise.grid(True, which="both", ls="-", alpha=0.5)
        fig_num_g_noise.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)

        # Save the new plot
        num_g_noise_plot_save_path = os.path.join(save_dir, "num_g_noise_impact_plot.png") # ★★★ 파일명 변경 ★★★
        try:
            fig_num_g_noise.savefig(num_g_noise_plot_save_path)
            print(f"Num_of_G with Noise impact plot saved to: {num_g_noise_plot_save_path}")
        except Exception as e:
            print(f"Error saving num_g with noise plot: {e}")

    else:
        print("Not enough data to plot num_of_G with Noise impact.")
        # fig_num_g_noise 변수가 없을 수 있으므로 None으로 설정
        fig_num_g_noise = None 
else:
    print(f"Skipping 'Impact of num_of_G with NOISE' experiment as config['noise_scaling_factor'] is {noise_factor_for_sweep}.")
    fig_num_g_noise = None # Ensure fig_num_g_noise exists for final save check if skipped


plt.show(block=True)
