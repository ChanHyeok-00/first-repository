#%%
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
import tempfile
import tensorflow_model_optimization as tfmot
import random

quantize_model = tfmot.quantization.keras.quantize_model

#%% 1. configuration
#===================================================================================================================
# 1. CONFIGURATION 
#===================================================================================================================

SEED = 42
#SEED = 777
#SEED = 2025
#SEED = 123456
#SEED = 987654
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

config = {
    "filepath": "/Users/ch/dataset/quant/Depression_voltage_arranged/V_D = -1.6 V", # conductance CSV 파일 경로
    "save_path": "~/python/quantization/results/", # 실험 데이터 저장 경로
    "mode": 3,                     # 전도도 데이터 가공 모드 (1, 2, 3)
    "clip": False,                 # 학습 시 가중치의 범위를 clip이 성능을 하락시켜서 비활성화 상태
    "use_scaling": True,           # 수 마이크로미터 단위의 전도도 값을 기존 모델의 단위(+_1)로 스케일링 할지
    "scaling_factor": 1e4,         # 전도도를 스케일링하는 정도
    "noise_scaling_factor": 40,    # 전도도 실험 데이터의 표준편차를 이용한 noise 구현, 노이즈의 크기를 정하는 변수
    "num_of_G": -1,                 # 총 100개의 전도도(평균) 값 중 몇 개를 사용할지 
    "G_sampling_method": "uniform_interval", # 총 100개의 전도도(평균) 값 중 사용할 값을 샘플링하는 방법 "uniform_interval" or "random", 성능은 둘다 비슷
    "epochs": 1,                   # epoch!
    "epochs_for_sweep": 0,         # 삭제 예정
    "batch_size": 64,              # batch size! 
    "apply_on_batch": False,       # QAT적용 방식, False: epoch마다, True: batch마다(mnist의 training data는 60,000개이므로 batch size가 64라면 한 번의 epoch마다 938번의 quantization을 수행하게 된다. 아래 변수를 조절해서 그 횟수를 줄일 수 있다.
    "qat_interval": 100,           # 100번의 batch마다 qat 적용
    "use_int_levels": False,       # 삭제 예정
    "int_min": -100,               # 삭제 예정
    "int_max": 100                 # 삭제 예정
}

# np.random.seed(21342145)

#%% 2. conductance manager
#===================================================================================================================
# 2. Conductance Manager
#===================================================================================================================

class ConductanceManager:
    '''
    전도도 데이터 불러오기
    전도도 데이터의 평균, 표준편차 구하기
    평균값은 가중치가 매핑할 값으로 사용, 표준편차는 노이즈로 사용
    데이터 스케일링, 전체 전도도 데이터 중 사용할 전도도의 개수만큼만 선택
    생성된 모델의 가중치를 전도도 값으로 매핑
    '''
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


    # 전도도를 불러오는 함수
    # 평균, 표준편차, 스케일링, 사용할 전도도 선택, 모드 선택
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
        print(f"Found {num_available_G_means} unique G_mean levels (raw pulses) in the CSV file.")


        if self.num_of_G <= 0 or self.num_of_G > num_available_G_means:
             self.num_of_G = num_available_G_means # Use all available if invalid num_of_G

        # 전도도 선택 방법 "random" or "uniform_interval"
        if self.G_sampling_method == "random":
            if self.num_of_G == num_available_G_means: # If asking for all, just take all
                 selected_G_means = available_G_means
                 selected_G_stds = available_G_stds
            elif self.num_of_G > 0 :
                random_indices = np.sort(np.random.choice(num_available_G_means, size=self.num_of_G, replace=False))
                selected_G_means = available_G_means[random_indices]
                selected_G_stds = available_G_stds[random_indices]
            else:
                selected_G_means = np.array([])
                selected_G_stds = np.array([])

        elif self.G_sampling_method == "uniform_interval":
            if self.num_of_G == 1 and num_available_G_means > 0:
                indices_to_select = np.array([num_available_G_means // 2], dtype=int)
            elif self.num_of_G > 0 and num_available_G_means > 0: # Ensure num_of_G > 0 for linspace
                indices_to_select = np.linspace(0, num_available_G_means - 1, num=self.num_of_G, dtype=int)
            else: 
                indices_to_select = np.array([], dtype=int)

            if indices_to_select.size > 0 :
                selected_G_means = available_G_means[indices_to_select]
                selected_G_stds = available_G_stds[indices_to_select]
            else:
                selected_G_means = np.array([])
                selected_G_stds = np.array([])
        else: # Default to selecting all if method is unknown or num_of_G is problematic
             selected_G_means = available_G_means
             selected_G_stds = available_G_stds
             if self.num_of_G > num_available_G_means : self.num_of_G = num_available_G_means


        print(f"Selected {len(selected_G_means)} positive conductance levels out of {num_available_G_means} available (Target: {self.num_of_G}, Method: {self.G_sampling_method}).")

        if len(selected_G_means) == 0: # Handle case where no levels are selected
            print("Warning: No conductance levels were selected. G_mean and G_std will be empty.")
            self.G_mean = np.array([])
            self.G_std = np.array([])
            self.min_weight = 0
            self.max_weight = 0
            return

        # 모드 선택
        # 모드1: 전도도 데이터 그대로 사용
        if self.mode == 1:
            G_mean_final_val = selected_G_means
            G_std_final_val = selected_G_stds
        # 모드2: 데이터의 최솟값이 0이 되도록 평행이동
        elif self.mode == 2:
            G_mean_final_val = selected_G_means - np.min(selected_G_means)
            G_std_final_val = selected_G_stds
        # 모드3: 데이터에 음수 부분 추가. (-G, 0, G)
        elif self.mode == 3:
            negative_part = -np.flip(selected_G_means)
            std_for_negative_part = np.flip(selected_G_stds)
            std_for_zero = np.min(selected_G_stds) if selected_G_stds.size > 0 else 0 
            G_mean_final_val = np.concatenate((negative_part, [0.0], selected_G_means))
            G_std_final_val = np.concatenate((std_for_negative_part, [std_for_zero], selected_G_stds))
            unique_means, unique_indices = np.unique(G_mean_final_val, return_index=True)
            G_mean_final_val = unique_means
            G_std_final_val = G_std_final_val[unique_indices]

        self.G_mean = G_mean_final_val
        self.G_std = G_std_final_val
        
        if self.G_mean.size == 0:
            print("Warning: G_mean is empty after processing. Setting min/max weights to 0.")
            self.min_weight = 0
            self.max_weight = 0
        else:
            print(f"[Conductance Levels] Count: {len(self.G_mean)}, Unique: {len(np.unique(self.G_mean))}")
            print(f"[G_mean] Min: {np.min(self.G_mean):.2e}, Max: {np.max(self.G_mean):.2e}")
            if self.G_std.size > 0 and self.G_std[self.G_std > 0].size > 0:
                 print(f"[G_std > 0] Min: {np.min(self.G_std[self.G_std > 0]):.2e}, Max: {np.max(self.G_std):.2e}")
            elif self.G_std.size > 0:
                 print(f"[G_std > 0] No G_std values greater than 0. Max G_std: {np.max(self.G_std):.2e}")
            else: 
                 print("[G_std > 0] G_std is empty.")

            if self.G_std.size == self.G_mean.size and self.G_std.size > 0 : 
                self.min_weight = np.min(self.G_mean - self.noise_scaling_factor * self.G_std)
                self.max_weight = np.max(self.G_mean + self.noise_scaling_factor * self.G_std)
            else: 
                self.min_weight = np.min(self.G_mean)
                self.max_weight = np.max(self.G_mean)


    # baseline 모델의 학습에서 가중치의 범위를 전도도 값의 최대, 최소로 제한하는 함수
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

    # 가중치를 전도도로 매핑하는 로직 함수
    def _quantize_value_to_g_mean(self, w_scalar):
        '''
        |전도도 - 가중치|값이 최소인 전도도 값의 인덱스 반환
        '''
        if self.G_mean is None or self.G_mean.size == 0: return w_scalar # Return original if no levels
        idx = np.argmin(np.abs(self.G_mean - w_scalar))
        return self.G_mean[idx]

    # 가중치를 전도도로 매핑하는 로직을 수행하는 함수
    def map_weights_to_g_mean(self, model):
        if self.G_mean is None or self.G_mean.size == 0:
            print("G_mean is not available. Skipping map_weights_to_g_mean.")
            return
        for layer in model.layers:
            if isinstance(layer, Dense):
                W, b = layer.get_weights()
                W_mapped = np.vectorize(self._quantize_value_to_g_mean)(W)
                layer.set_weights([W_mapped, b])
            elif isinstance(layer, tfmot.quantization.keras.QuantizeWrapperV2):
                print(f"Skipping QuantizeWrapperV2 layer: {layer.name} in map_weights_to_g_mean (expected for framework QAT)")

    # 노이즈를 추가하는 함수
    def add_noise_to_weights(self, model):
        if self.noise_scaling_factor == 0:
            print("Noise scaling factor is 0, no noise will be added.")
            return
        if self.G_mean is None or self.G_mean.size == 0 or self.G_std is None or self.G_std.size == 0 or self.G_mean.shape != self.G_std.shape:
            print("G_mean or G_std is not available, or shapes mismatch. Skipping add_noise_to_weights.")
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
                # print(f"\nCustom QAT callback in {context_str}:") 
                if self.cm.G_mean is not None and self.cm.G_mean.size > 0:
                    # print(f"  Mapping weights to G_mean...") 
                    self.cm.map_weights_to_g_mean(self.target_model)
                    if self.inject_noise and self.cm.noise_scaling_factor > 0:
                        # print(f"  Adding noise to weights...") 
                        self.cm.add_noise_to_weights(self.target_model)
                else:
                    print("  G_mean not available for QAT. Skipping QAT operations.")


            def on_epoch_end(self, epoch, logs=None):
                if not self.apply_on_batch:
                    self._perform_qat_operations(context_str=f"end of epoch {epoch+1}")

            def on_train_batch_end(self, batch, logs=None):
                if self.apply_on_batch:
                    if (batch + 1) % config_param.get("qat_interval", 100) == 0: 
                        self._perform_qat_operations(context_str=f"end of batch {batch+1}")
        return CustomQATCallback(self, model_to_modify, inject_noise_during_training, apply_on_batch)
    
    # 가중치가 매핑이 이루어졌는지 확인하는 함수
    def check_weights_mapped(self, model, tolerance=1e-6):
        if self.G_mean is None or self.G_mean.size == 0:
            print("\nChecking weights mapped: G_mean is not set or is empty. Cannot check mapping.")
            return

        print("\nChecking if model weights are mapped to G_mean levels:")
        all_mapped = True
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Dense):
                weights_data = layer.get_weights()
                if not weights_data : 
                    print(f"Layer {i} ({layer.name}): No weights found to check.")
                    continue
                weights = weights_data[0].flatten()
                mismatched_count = 0
                for w in weights:
                    if not np.any(np.isclose(w, self.G_mean, rtol=0, atol=tolerance)):
                        mismatched_count += 1
                if mismatched_count > 0:
                    print(f"Layer {i} ({layer.name}): {mismatched_count}/{len(weights)} weights NOT mapped (tolerance={tolerance:.1e}).")
                    all_mapped = False
                else:
                    print(f"Layer {i} ({layer.name}): All {len(weights)} weights mapped (tolerance={tolerance:.1e}).")
        if all_mapped: print("All Dense layer weights appear to be mapped correctly.")
        else: print("Some Dense layer weights are not mapped to G_mean (or outside tolerance).")


#%% 3. helper functions
#===================================================================================================================
# 3. HELPER FUNCTIONS (Callbacks, Model Def, Data, Evaluation, Plotting)
#===================================================================================================================
class BatchMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, print_batch_loss_freq=0):
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
        if logs is not None:
            self.batch_losses.append(logs.get('loss'))
            if self.print_batch_loss_freq > 0 and (batch + 1) % self.print_batch_loss_freq == 0:
                print(f" - Batch {batch+1}: loss = {logs.get('loss'):.4f}")
    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self, 'params') and self.params:
            print(f"Epoch {epoch+1}/{self.params.get('epochs', 'N/A')}")
        else:
            print(f"Epoch {epoch+1}/N/A (params not set in callback yet)")
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.epoch_train_losses.append(logs.get('loss'))
            self.epoch_train_accuracies.append(logs.get('accuracy'))
            self.epoch_val_losses.append(logs.get('val_loss'))
            self.epoch_val_accuracies.append(logs.get('val_accuracy'))
            print(f"Epoch {epoch+1}: loss={logs.get('loss'):.4f}, acc={logs.get('accuracy'):.4f}, val_loss={logs.get('val_loss'):.4f}, val_acc={logs.get('val_accuracy'):.4f}")

def create_mnist_model(kernel_constraint=None, input_shape=(28, 28)):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    model = Sequential([
        Input(shape=input_shape), Flatten(),
        Dense(128, activation="relu", kernel_initializer=initializer, kernel_constraint=kernel_constraint),
        Dense(64, activation="relu", kernel_initializer=initializer, kernel_constraint=kernel_constraint),
        Dense(10, activation="softmax", kernel_initializer=initializer, kernel_constraint=kernel_constraint)
    ])
    return model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
    yield [input_value]

def evaluate_model(model, model_name, x_test_data, y_test_data, is_tflite=False, tflite_interpreter=None):
    print(f"\n--- Evaluating: {model_name} ---")
    if is_tflite:
        if tflite_interpreter is None: print("Error: TFLite interpreter not provided."); return 0.0, 0.0
        input_details = tflite_interpreter.get_input_details(); output_details = tflite_interpreter.get_output_details()
        if not input_details or not output_details: print(f"Error: Could not get input/output details for TFLite model {model_name}"); return 0.0,0.0
        input_dtype = input_details[0]['dtype']; input_quant_params = input_details[0]['quantization_parameters']
        input_scale_list = input_quant_params.get('scales', []); input_zp_list = input_quant_params.get('zero_points', [])
        input_scale = input_scale_list[0] if input_scale_list and len(input_scale_list) > 0 else 0.0
        input_zero_point = input_zp_list[0] if input_zp_list and len(input_zp_list) > 0 else 0
        is_input_quantized = (input_dtype == np.int8 or input_dtype == np.uint8) and input_scale != 0.0
        output_dtype = output_details[0]['dtype']; output_quant_params = output_details[0]['quantization_parameters']
        output_scale_list = output_quant_params.get('scales', []); output_zp_list = output_quant_params.get('zero_points', [])
        output_scale = output_scale_list[0] if output_scale_list and len(output_scale_list) > 0 else 0.0
        output_zero_point = output_zp_list[0] if output_zp_list and len(output_zp_list) > 0 else 0
        is_output_quantized = (output_dtype == np.int8 or output_dtype == np.uint8) and output_scale != 0.0
        if is_input_quantized: print(f"[INFO] {model_name}: Input requires {input_dtype}. Scale={input_scale:.4e}, ZeroPoint={input_zero_point}")
        if is_output_quantized: print(f"[INFO] {model_name}: Output is {output_dtype}. Scale={output_scale:.4e}, ZeroPoint={output_zero_point}")
        num_correct = 0; num_total = len(x_test_data)
        for i in range(num_total):
            float_image_batch = np.expand_dims(x_test_data[i], axis=0).astype(np.float32)
            if is_input_quantized:
                if input_scale == 0: print(f"Warning: Input scale is 0 for {model_name}. Cannot quantize input."); processed_input = float_image_batch.astype(input_dtype)
                else:
                    quantized_image_batch = (float_image_batch / input_scale) + input_zero_point
                    quantized_image_batch = np.round(quantized_image_batch)
                    if input_dtype == np.int8: quantized_image_batch = np.clip(quantized_image_batch, -128, 127)
                    elif input_dtype == np.uint8: quantized_image_batch = np.clip(quantized_image_batch, 0, 255)
                    processed_input = quantized_image_batch.astype(input_dtype)
            else: processed_input = float_image_batch.astype(input_dtype)
            final_input = processed_input; expected_shape = tuple(input_details[0]['shape']); current_shape = processed_input.shape
            if len(expected_shape) == 4 and expected_shape[3] == 1 and len(current_shape) == 3: final_input = np.expand_dims(final_input, axis=-1)
            elif len(expected_shape) == 2 and expected_shape[0] == 1 and len(current_shape) > 1 and expected_shape[1] == np.prod(current_shape[1:]): final_input = final_input.reshape(1, -1)
            tflite_interpreter.set_tensor(input_details[0]['index'], final_input); tflite_interpreter.invoke()
            output_data = tflite_interpreter.get_tensor(output_details[0]['index'])
            if is_output_quantized:
                if output_scale == 0: print(f"Warning: Output scale is 0 for {model_name}. Cannot dequantize output."); output_data_float = output_data.astype(np.float32)
                else: output_data_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
            else: output_data_float = output_data
            if np.argmax(output_data_float) == y_test_data[i]: num_correct += 1
        accuracy = num_correct / num_total if num_total > 0 else 0.0
        print(f"Test Accuracy: {accuracy:.4f} (Note: Loss for TFLite models is not calculated here)"); return 0.0, accuracy 
    else: 
        loss, accuracy = model.evaluate(x_test_data, y_test_data, verbose=0)
        print(f"Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}"); return loss, accuracy

def print_weight_statistics(model, name):
    print(f"\nWeight Statistics for {name}:")
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            weights_list = layer.get_weights()
            if weights_list: weights = weights_list[0]; print(f" Layer {i} ({layer.name}): Min={np.min(weights):.4f}, Max={np.max(weights):.4f}, Mean={np.mean(weights):.4f}, Std={np.std(weights):.4f}")
            else: print(f" Layer {i} ({layer.name}): No weights found.")
        elif isinstance(layer, tfmot.quantization.keras.QuantizeWrapperV2):
            try:
                unquantized_weights = None
                for weight_var in layer.trainable_weights:
                    if 'kernel' in weight_var.name and 'quantize_layer' not in weight_var.name: unquantized_weights = weight_var.numpy(); break
                if unquantized_weights is not None: print(f" Layer {i} ({layer.name} - QAT Float Kernel): Min={np.min(unquantized_weights):.4f}, Max={np.max(unquantized_weights):.4f}, Mean={np.mean(unquantized_weights):.4f}, Std={np.std(unquantized_weights):.4f}")
                else: print(f" Layer {i} ({layer.name} - QAT): Could not extract comparable float kernel weight stats.")
            except Exception as e: print(f" Layer {i} ({layer.name} - QAT): Error extracting weight stats: {e}")

def plot_metric_histogram(data, title, xlabel, ax=None, bins=50):
    if ax is None: fig_standalone, current_ax = plt.subplots(figsize=(8, 5))
    else: current_ax = ax
    if not data: current_ax.text(0.5, 0.5, "No data for histogram", ha='center', va='center', transform=current_ax.transAxes)
    else:
        data_array = np.array(data)
        if data_array.size == 0 : current_ax.text(0.5, 0.5, "No data for histogram (empty array)", ha='center', va='center', transform=current_ax.transAxes)
        else: current_ax.hist(data_array, bins=bins, edgecolor='black')
    current_ax.set_title(title, fontsize=10); current_ax.set_xlabel(xlabel, fontsize=8); current_ax.set_ylabel("Frequency", fontsize=8)
    current_ax.tick_params(axis='both', which='major', labelsize=7); current_ax.grid(axis='y', alpha=0.75)
    if ax is None: plt.show(block=False)

def plot_weight_hist(model, title="Weight Distribution", bins=50, ax=None, tflite_interpreter=None, plot_dequantized=False):
    current_ax = ax
    if current_ax is None: fig_standalone, current_ax = plt.subplots(figsize=(8, 5))
    w_all = []; x_label = "Weight Value"; is_tflite_model = model is None and tflite_interpreter is not None
    if is_tflite_model:
        tensor_dtype_str = "unknown"
        if not tflite_interpreter.get_output_details(): current_ax.text(0.5, 0.5, "TFLite output details missing", ha='center', va='center'); return
        first_output_idx = tflite_interpreter.get_output_details()[0]['index']
        all_tensor_details = tflite_interpreter.get_tensor_details()
        # print("\n--- TFLite 모델 내부 텐서 정보 (디버깅용) ---") 
        # for i, details in enumerate(all_tensor_details): print(f"텐서 #{i}: {details}")
        # print("------------------------------------------\n")
        for tensor_details in all_tensor_details:
            is_weight_tensor = (tensor_details.get('dtype') == np.int8 and len(tensor_details.get('shape', [])) > 1 and 'bias' not in tensor_details.get('name', '') and (tensor_details.get('index', float('inf')) < first_output_idx))
            if is_weight_tensor : 
                tensor_data = tflite_interpreter.get_tensor(tensor_details['index']); tensor_dtype_str = str(tensor_data.dtype)
                if tensor_data.ndim > 0:
                    if plot_dequantized and tensor_details.get('quantization_parameters') and tensor_details['quantization_parameters'].get('scales') and len(tensor_details['quantization_parameters']['scales']) > 0 and tensor_details['quantization_parameters'].get('zero_points') and len(tensor_details['quantization_parameters']['zero_points']) > 0:
                        scale = tensor_details['quantization_parameters']['scales'][0]; zero_point = tensor_details['quantization_parameters']['zero_points'][0]
                        if scale != 0: dequantized_data = (tensor_data.astype(np.float32) - zero_point) * scale; w_all.extend(dequantized_data.flatten())
                        else: w_all.extend(tensor_data.flatten())
                    else: w_all.extend(tensor_data.flatten())
        x_label = f"Weight Value ({'Dequantized F32' if plot_dequantized and w_all else 'Raw ' + tensor_dtype_str})"
    elif model is not None: 
        for layer in model.layers:
            layer_kernel_weights = None
            if isinstance(layer, tfmot.quantization.keras.QuantizeWrapperV2):
                try:
                    float_kernel_candidates = [w for w in layer.trainable_weights if 'kernel' in w.name and 'quantizer' not in w.name.lower()]
                    if float_kernel_candidates: layer_kernel_weights = float_kernel_candidates[0].numpy().flatten()
                    elif hasattr(layer.layer, 'kernel'): layer_kernel_weights = layer.layer.kernel.numpy().flatten()
                except AttributeError: pass
            elif isinstance(layer, tf.keras.layers.Dense):
                if layer.get_weights(): layer_kernel_weights = layer.get_weights()[0].flatten()
            if layer_kernel_weights is not None: w_all.extend(layer_kernel_weights)
        x_label = "Weight Value (Float32 from Keras)"
    if not w_all: current_ax.text(0.5, 0.5, "No weights found/extracted", ha='center', va='center', transform=current_ax.transAxes)
    else: current_ax.hist(np.array(w_all), bins=bins, edgecolor='black')
    current_ax.set_title(title, fontsize=10); current_ax.set_xlabel(x_label, fontsize=8); current_ax.set_ylabel("Count", fontsize=8)
    current_ax.tick_params(axis='both', which='major', labelsize=7); current_ax.grid(axis='y', alpha=0.75)
    if ax is None: plt.show(block=False)

# --- Main Experiment ---
results = {}
model_history_callbacks = {} 
TOTAL_MAIN_PLOTS = 20 
NUM_ROWS = 5 
NUM_COLS = 3 
fig_main, all_axes_main = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(14, 15))
axes_flat_main = all_axes_main.flatten()
plot_idx_main = 0
conductance_handler = ConductanceManager(
    filepath=config["filepath"], mode=config["mode"], clip=config["clip"],
    use_scaling=config["use_scaling"], scaling_factor=config["scaling_factor"],
    noise_scaling_factor=config["noise_scaling_factor"],
    use_int_levels=config["use_int_levels"], int_min=config["int_min"], int_max=config["int_max"],
    num_of_G=config.get("num_of_G", -1), G_sampling_method=config.get("G_sampling_method", "uniform_interval"),
    apply_on_batch=config.get("apply_on_batch", False)
)

def record_initial_perf(model, model_name_suffix, x_data, y_data, results_dict):
    loss, acc = model.evaluate(x_data, y_data, verbose=0)
    key = f"Initial_{model_name_suffix}"
    results_dict[key] = {"loss": loss, "accuracy": acc}
    print(f"{key}: Loss={loss:.4f}, Accuracy={acc:.4f}")

def plot_metric_history(history_callback, metric_plot_configs, ax, title_prefix=""):
    epochs_ran = 0
    if history_callback and hasattr(history_callback, 'epoch_train_losses') and history_callback.epoch_train_losses:
        epochs_ran = len(history_callback.epoch_train_losses)
    if epochs_ran == 0:
        ax.text(0.5, 0.5, "No training data for history plot", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title_prefix}History (No Data)", fontsize=9); return
    epoch_range = range(1, epochs_ran + 1); combined_labels = []; y_label_parts = []
    for data_key, label, color in metric_plot_configs:
        metric_data = getattr(history_callback, data_key, [])
        if metric_data and len(metric_data) == epochs_ran:
            ax.plot(epoch_range, metric_data, marker='.', linestyle='-', label=label, color=color); combined_labels.append(label)
            if "Acc" in label and "Accuracy" not in y_label_parts: y_label_parts.append("Accuracy")
            if "Loss" in label and "Loss" not in y_label_parts: y_label_parts.append("Loss")
        else: print(f"[plot_metric_history] Data missing or mismatched for {data_key} (expected {epochs_ran}, got {len(metric_data) if metric_data else 0})")
    if combined_labels:
        ax.set_title(f"{title_prefix}{'/'.join(combined_labels)} vs Epochs", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8); ax.set_ylabel(' / '.join(y_label_parts) if y_label_parts else "Metric Value", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        if len(combined_labels) > 0: ax.legend(fontsize=7)
        ax.grid(True, alpha=0.6)
    else:
        ax.text(0.5, 0.5, "No data for selected metrics", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title_prefix}History (Metrics Missing)", fontsize=9)

#%% 4. execution
#===================================================================================================================
# 4. EXPERIMENT EXECUTION
#===================================================================================================================

# --- 1. Baseline Float32 Model ---
print("\n=== 1. Baseline Float32 Model ===")
model_baseline = create_mnist_model(kernel_constraint=conductance_handler.get_kernel_constraint())
model_baseline.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
record_initial_perf(model_baseline, "Baseline_Float32", x_test, y_test, results)
baseline_callbacks = [BatchMetricsCallback(print_batch_loss_freq=0)]
history_baseline = model_baseline.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_data=(x_test,y_test), callbacks=baseline_callbacks, verbose=0)
test_loss_bl, test_acc_bl = evaluate_model(model_baseline, "Baseline Float32 (Trained)", x_test, y_test)
results["Baseline_Float32_Trained"] = {
    "train_loss": baseline_callbacks[0].epoch_train_losses[-1] if baseline_callbacks[0].epoch_train_losses else np.nan, 
    "train_accuracy": baseline_callbacks[0].epoch_train_accuracies[-1] if baseline_callbacks[0].epoch_train_accuracies else np.nan,
    "val_loss": baseline_callbacks[0].epoch_val_losses[-1] if baseline_callbacks[0].epoch_val_losses else np.nan, 
    "val_accuracy": baseline_callbacks[0].epoch_val_accuracies[-1] if baseline_callbacks[0].epoch_val_accuracies else np.nan,
    "test_loss": test_loss_bl, "test_accuracy": test_acc_bl
}
model_history_callbacks["Baseline_Float32"] = baseline_callbacks[0]
print_weight_statistics(model_baseline, "Baseline Float32 (Trained)")
if plot_idx_main < len(axes_flat_main): plot_weight_hist(model_baseline, "1. Baseline Float32 Weights", bins=100, ax=axes_flat_main[plot_idx_main]); plot_idx_main += 1

# --- 2. Custom PTQ (G_mean mapping Only) ---
print("\n=== 2. Custom PTQ (G_mean mapping Only) ===")
model_ptq_g_mean_only = tf.keras.models.clone_model(model_baseline); model_ptq_g_mean_only.set_weights(model_baseline.get_weights())
model_ptq_g_mean_only.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("Mapping weights to G_mean (NO noise)..."); 
original_noise_factor_ptq_gmo = conductance_handler.noise_scaling_factor # 원래 noise factor 저장
conductance_handler.noise_scaling_factor = 0 # PTQ G-mean only 시에는 noise factor 0으로 설정
conductance_handler.map_weights_to_g_mean(model_ptq_g_mean_only)
conductance_handler.noise_scaling_factor = original_noise_factor_ptq_gmo # 원래 noise factor 복원
loss_ptq_gmo, acc_ptq_gmo = evaluate_model(model_ptq_g_mean_only, "Custom PTQ (G_mean Only)", x_test, y_test)
results["Custom_PTQ_G_mean_Only_Mapped"] = {"test_loss": loss_ptq_gmo, "test_accuracy": acc_ptq_gmo} # PTQ는 test 결과만 저장
print_weight_statistics(model_ptq_g_mean_only, "Custom PTQ (G_mean Only)"); conductance_handler.check_weights_mapped(model_ptq_g_mean_only, tolerance=1e-7)
if plot_idx_main < len(axes_flat_main): plot_weight_hist(model_ptq_g_mean_only, f"2. Custom PTQ G_mean Only", bins=100, ax=axes_flat_main[plot_idx_main]); plot_idx_main += 1

# --- 3. Custom QAT (G_mean mapping + Config Noise during training) ---
print("\n=== 3. Custom QAT (G_mean mapping + Config Noise during training) ===") 
if config.get("noise_scaling_factor", 0) > 0:
    conductance_handler.noise_scaling_factor = config["noise_scaling_factor"]
    model_qat_custom_noise_train = create_mnist_model(kernel_constraint=conductance_handler.get_kernel_constraint())
    model_qat_custom_noise_train.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    record_initial_perf(model_qat_custom_noise_train, "Custom_QAT_G_mean_Config_Noise", x_test, y_test, results)
    custom_qat_noise_cb = conductance_handler.get_custom_qat_callback(model_qat_custom_noise_train, config, inject_noise_during_training=True)
    qat_noise_config_hist_cb = BatchMetricsCallback(print_batch_loss_freq=0)
    model_qat_custom_noise_train.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_data=(x_test, y_test), callbacks=[custom_qat_noise_cb, qat_noise_config_hist_cb], verbose=0)
    
    final_train_loss_cn = qat_noise_config_hist_cb.epoch_train_losses[-1] if qat_noise_config_hist_cb.epoch_train_losses else np.nan
    final_train_acc_cn = qat_noise_config_hist_cb.epoch_train_accuracies[-1] if qat_noise_config_hist_cb.epoch_train_accuracies else np.nan
    final_val_loss_cn = qat_noise_config_hist_cb.epoch_val_losses[-1] if qat_noise_config_hist_cb.epoch_val_losses else np.nan
    final_val_acc_cn = qat_noise_config_hist_cb.epoch_val_accuracies[-1] if qat_noise_config_hist_cb.epoch_val_accuracies else np.nan
    test_loss_cn, test_acc_cn = evaluate_model(model_qat_custom_noise_train, "Custom QAT (G_mean + Config Noise Trained)", x_test, y_test)
    
    model_name_key_cn = "Custom_QAT_G_mean_Config_Noise_Trained"
    results[model_name_key_cn] = {
        "train_loss": final_train_loss_cn, "train_accuracy": final_train_acc_cn,
        "val_loss": final_val_loss_cn, "val_accuracy": final_val_acc_cn,
        "test_loss": test_loss_cn, "test_accuracy": test_acc_cn
    }
    print(f"  Final {model_name_key_cn} Metrics:")
    print(f"    Train - Loss: {final_train_loss_cn:.4f}, Accuracy: {final_train_acc_cn:.4f}")
    print(f"    Val   - Loss: {final_val_loss_cn:.4f}, Accuracy: {final_val_acc_cn:.4f}")
    print(f"    Test  - Loss: {test_loss_cn:.4f}, Accuracy: {test_acc_cn:.4f}")
    model_history_callbacks["Custom_QAT_G_mean_Config_Noise"] = qat_noise_config_hist_cb
    print_weight_statistics(model_qat_custom_noise_train, "Custom QAT (G_mean + Config Noise Trained)")
    conductance_handler.check_weights_mapped(model_qat_custom_noise_train, tolerance=conductance_handler.noise_scaling_factor * 1e-1 if conductance_handler.noise_scaling_factor > 0 else 1e-5)
    if plot_idx_main < len(axes_flat_main): plot_weight_hist(model_qat_custom_noise_train, f"3. Custom QAT G_mean + Config Noise", bins=100, ax=axes_flat_main[plot_idx_main]); plot_idx_main += 1
else:
    print(f"Skipping Custom QAT with Config Noise as config[\"noise_scaling_factor\"] is {config.get('noise_scaling_factor', 0)}.")
    results["Custom_QAT_G_mean_Config_Noise_Trained"] = {"train_loss": np.nan, "train_accuracy": np.nan, "val_loss": np.nan, "val_accuracy": np.nan, "test_loss": np.nan, "test_accuracy": np.nan, "error": "Skipped"}
    if plot_idx_main < len(axes_flat_main): axes_flat_main[plot_idx_main].text(0.5, 0.5, "Skipped: Custom QAT with Config Noise", ha='center', va='center', fontsize=9); axes_flat_main[plot_idx_main].set_title("3. QAT G-mean + Config Noise (Skipped)", fontsize=9); plot_idx_main += 1


# --- 4. Custom PTQ (G_mean mapping + Noise) ---
print("\n=== 4. Custom PTQ (G_mean mapping + Noise) ===") 
conductance_handler.noise_scaling_factor = config["noise_scaling_factor"]
model_ptq_custom_noise = tf.keras.models.clone_model(model_baseline) 
model_ptq_custom_noise.set_weights(model_baseline.get_weights())
model_ptq_custom_noise.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("Mapping weights to G_mean for PTQ with Noise..."); conductance_handler.map_weights_to_g_mean(model_ptq_custom_noise)
if conductance_handler.noise_scaling_factor > 0: 
    print("Adding noise based on G_std for PTQ with Noise..."); conductance_handler.add_noise_to_weights(model_ptq_custom_noise)
else:
    print("Noise scaling factor is 0, skipping noise addition for PTQ (Noise experiment).")
loss, acc = evaluate_model(model_ptq_custom_noise, "Custom PTQ (G_mean + Noise)", x_test, y_test)
results["Custom_PTQ_G_mean_Noise_Mapped"] = {"test_loss": loss, "test_accuracy": acc} # PTQ는 test 결과만 저장
print_weight_statistics(model_ptq_custom_noise, "Custom PTQ (G_mean + Noise)")
conductance_handler.check_weights_mapped(model_ptq_custom_noise, tolerance=conductance_handler.noise_scaling_factor * 1e-2 if conductance_handler.noise_scaling_factor > 0 else 1e-7)
if plot_idx_main < len(axes_flat_main):
    plot_weight_hist(model_ptq_custom_noise, f"5. Custom PTQ G_mean + Noise", bins=100, ax=axes_flat_main[plot_idx_main])
    plot_idx_main += 1

# Add training history plots
def add_history_plots(model_key_name, title_prefix_str):
    global plot_idx_main 
    cb_data = model_history_callbacks.get(model_key_name)
    plots_to_create = 3 if model_key_name == "Baseline_Float32" else 2 # Baseline은 BatchLossHist 포함
    
    if cb_data:
        # Accuracy Plot
        if plot_idx_main < len(axes_flat_main):
            plot_metric_history(cb_data, [('epoch_train_accuracies', 'Train Acc', 'tab:blue'), ('epoch_val_accuracies', 'Val Acc', 'tab:orange')], ax=axes_flat_main[plot_idx_main], title_prefix=title_prefix_str)
            plot_idx_main += 1
        else: print(f"Not enough subplot space for {title_prefix_str} Acc History")
        # Loss Plot
        if plot_idx_main < len(axes_flat_main):
            plot_metric_history(cb_data, [('epoch_train_losses', 'Train Loss', 'tab:green'), ('epoch_val_losses', 'Val Loss', 'tab:red')], ax=axes_flat_main[plot_idx_main], title_prefix=title_prefix_str)
            plot_idx_main += 1
        else: print(f"Not enough subplot space for {title_prefix_str} Loss History")
        # Batch Loss Histogram for Baseline
        if model_key_name == "Baseline_Float32":
            if plot_idx_main < len(axes_flat_main):
                if cb_data.batch_losses:
                    plot_metric_histogram(cb_data.batch_losses, f"{title_prefix_str}Batch Losses Dist.", "Batch Loss Value", ax=axes_flat_main[plot_idx_main], bins=50)
                else: # 데이터는 있지만 batch_losses가 비어있는 경우
                    axes_flat_main[plot_idx_main].text(0.5, 0.5, "No Batch Loss Data", ha='center', va='center', fontsize=9)
                    axes_flat_main[plot_idx_main].set_title(f"{title_prefix_str}Batch Losses (No Data)", fontsize=9)
                plot_idx_main += 1
            else: print(f"Not enough subplot space for {title_prefix_str} Batch Loss Histogram")
    else:
        print(f"{model_key_name} callback data not found for history plots.")
        for i in range(plots_to_create): 
            if plot_idx_main < len(axes_flat_main):
                axes_flat_main[plot_idx_main].text(0.5, 0.5, "Plot Skipped\n(No History Data)", ha='center', va='center', fontsize=9)
                axes_flat_main[plot_idx_main].set_title(f"{title_prefix_str}History (Skipped)", fontsize=9)
                plot_idx_main += 1

add_history_plots("Baseline_Float32", "Baseline: ")
add_history_plots("Custom_QAT_G_mean_Config_Noise", "QAT ConfigNoise: ")


# --- PTQ G-mean Only result on Baseline Acc plot ---
baseline_cb_data = model_history_callbacks.get("Baseline_Float32")
target_baseline_acc_ax = None
# Baseline Acc 플롯의 제목을 정확히 찾아야 함 (add_history_plots 함수에서 설정된 제목 기준)
baseline_acc_plot_title_search = "Baseline: Train Acc/Val Acc vs Epochs" 
for ax_search in axes_flat_main[:plot_idx_main]: # 이미 그려진 플롯들 중에서만 검색
    if ax_search.get_title() == baseline_acc_plot_title_search:
        target_baseline_acc_ax = ax_search
        break

if baseline_cb_data and target_baseline_acc_ax:
    ptq_result = results.get("Custom_PTQ_G_mean_Only_Mapped")
    if ptq_result and baseline_cb_data.epoch_val_accuracies: 
        final_epoch = len(baseline_cb_data.epoch_val_accuracies)
        if final_epoch > 0: 
            final_acc = ptq_result.get('test_accuracy', np.nan) # PTQ 결과는 test_accuracy 사용
            if not np.isnan(final_acc):
                target_baseline_acc_ax.scatter(final_epoch, final_acc, marker='*', s=200, color='red', zorder=10, label=f'PTQ G-mean Acc ({final_acc:.4f})')
                target_baseline_acc_ax.legend()
else:
    if not baseline_cb_data : print("PTQ star marker: Baseline callback data not found.")
    if not target_baseline_acc_ax : print("PTQ star marker: Target baseline acc axis not found or title mismatch.")


# --- Clean up unused subplots in the main figure ---
for i in range(plot_idx_main, NUM_ROWS * NUM_COLS):
    if i < len(axes_flat_main): 
      fig_main.delaxes(axes_flat_main[i])
fig_main.suptitle(f"MNIST Quantization Experiments (G_levels={config.get('num_of_G','N/A')}, NoiseFactor={config.get('noise_scaling_factor','N/A')})", fontsize=16)
fig_main.tight_layout(rect=[0, 0.03, 1, 0.96]) 

# --- Final Results Summary (Text) ---
print("\n\n--- Experiment Summary (Main Set) ---")
print(f"Global SEED: {SEED}")
print(f"Config num_of_G: {config.get('num_of_G', 'N/A')}, Config noise_scaling_factor: {config.get('noise_scaling_factor', 'N/A')}")
print(f"Config epochs: {config.get('epochs', 'N/A')}, Config epochs_for_sweep: {config.get('epochs_for_sweep', 'N/A')}")

sorted_results_keys = sorted(results.keys())
for model_name in sorted_results_keys:
    metrics = results[model_name]
    print(f"\n{model_name}:")
    if "error" in metrics:
        print(f"  Error: {metrics['error']}")
        if "test_accuracy" in metrics: print(f"  Test Accuracy: {metrics.get('test_accuracy', 'N/A')}") # 오류가 있어도 정확도/손실이 있을 수 있음
        if "test_loss" in metrics: print(f"  Test Loss: {metrics.get('test_loss', 'N/A')}")
    elif "test_accuracy" in metrics: # QAT 모델 또는 PTQ 모델
        if "train_accuracy" in metrics : # QAT 스타일
            print(f"  Train Metrics: Loss = {metrics.get('train_loss', np.nan):.4f}, Accuracy = {metrics.get('train_accuracy', np.nan):.4f}")
            print(f"  Val. Metrics : Loss = {metrics.get('val_loss', np.nan):.4f}, Accuracy = {metrics.get('val_accuracy', np.nan):.4f}")
        # 모든 경우에 Test Metrics 출력 (PTQ는 이것만 있을 수 있음)
        print(f"  Test Metrics : Loss = {metrics.get('test_loss', np.nan):.4f}, Accuracy = {metrics.get('test_accuracy', np.nan):.4f}")
    else: # 초기값 등 test_accuracy 키가 없는 경우
        print(f"  Loss: {metrics.get('loss', 'N/A')}, Accuracy: {metrics.get('accuracy', 'N/A')}")


# --- Save results (Config, JSON, Plots) ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_save_dir = os.path.expanduser(config.get("save_path"))
os.makedirs(base_save_dir, exist_ok=True)
experiment_name_parts = [
    "mnist", f"mode{config['mode']}", f"g{config.get('num_of_G','N/A')}", 
    f"noise{config.get('noise_scaling_factor','N/A')}", timestamp
]
save_dir_name = "_".join(experiment_name_parts)
save_dir = os.path.join(base_save_dir, save_dir_name)
os.makedirs(save_dir, exist_ok=True)

config_save_path = os.path.join(save_dir, "config.json")
results_save_path = os.path.join(save_dir, "results.json")
main_plot_save_path = os.path.join(save_dir, "main_plots.png")

with open(config_save_path, "w") as f:
    serializable_config = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in config.items()}
    json.dump(serializable_config, f, indent=4)
with open(results_save_path, "w") as f:
    serializable_results = {}
    for k, v_dict in results.items():
        serializable_results[k] = { sub_k: (float(sub_v) if isinstance(sub_v, (np.floating, float)) and np.isfinite(sub_v) else str(sub_v)) for sub_k, sub_v in v_dict.items()}
    json.dump(serializable_results, f, indent=4)

print(f"\nConfiguration and results saved to: {save_dir}")

if 'fig_main' in locals() and fig_main is not None:
    try: fig_main.savefig(main_plot_save_path); print(f"Main experiment plots saved to: {main_plot_save_path}")
    except Exception as e: print(f"Error saving main plot: {e}")

plt.show(block=True)
