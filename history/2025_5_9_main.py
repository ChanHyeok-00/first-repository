#%% 학습
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
# plt.ion()

# config
config = {
    "filepath": "/Users/ch/dataset/quant/Depression_voltage_arranged/V_D = -1.6 V",
    "mode": 3,
    "clip": True,
    "G_scaling": True,
    "scaling_factor": 2e4,
    "noise_scaling_factor": 1,
    "round_G_mean": False,
    "use_ptq": True,
    "use_qat": True
}


# 모델 클래스
class Model:
    def __init__(self, filepath, mode=1, clip=False, G_scaling=False, scaling_factor=1, noise_scaling_factor=0, use_qat=False, use_ptq=False, round_G_mean=False):
        self.filepath = filepath
        self.mode = mode
        self.use_clip = clip
        self.use_scaling = G_scaling
        self.scaling_factor = scaling_factor
        self.noise_scaling_factor = noise_scaling_factor
        self.round_G_mean = round_G_mean
        self.use_ptq = use_ptq
        self.use_qat = use_qat

        self.G = None
        self.G_mean = None
        self.G_std = None
        # self.Gn_mean = None
        # self.Gn_std = None
        self.min_weight = None
        self.max_weight = None

        self._load_conductance()

        
    # 가중치로 사용할 전도도를 다루는 함수
    def _load_conductance(self):
        df = pd.read_csv(self.filepath, sep='\t', header=None)
        G = df.to_numpy()

        if self.use_scaling:
            G = G * self.scaling_factor

        G_stats = np.array([[np.mean(row[1:]), np.std(row[1:])] for row in G])
        G_mean = G_stats[:, 0]
        G_std = G_stats[:, 1]

        # 전도도 값의 유효숫자가 1개가 되도록 양자화
        if self.round_G_mean:
            # 1) 각 값의 지수(magnitude)와 단위(unit) 계산
            mags  = np.floor(np.log10(np.abs(G_mean)))
            units = np.power(10.0, mags)

            # 2) 절댓값을 단위로 나눈 뒤 반올림하여 정수 레벨 획득
            rounded_abs = np.round(np.abs(G_mean) / units).astype(int)

            # 3) 실제값 계산, 중복 제거
            rounded_levels = rounded_abs * units
            unique_levels  = np.unique(rounded_levels)

            # 4) 각 실제 레벨에 대응하는 원본 G_std 평균 계산
            std_levels = np.array([
                np.mean(G_std[np.isclose(rounded_levels, lvl)])
                for lvl in unique_levels
            ])

            # 5) G_mean/G_std 덮어쓰기
            G_mean = unique_levels
            G_std  = std_levels

        # 가중치 변형 모드 1~5
        if self.mode == 1:
            pass

        elif self.mode == 2:

            G_mean -= np.min(G_mean)

        elif self.mode == 3:
            G_mean = np.concatenate([-G_mean, [0], G_mean])
            G_std = np.concatenate([G_std, [0], G_std])

        '''elif self.mode == 4:
            G_mean -= np.min(G_mean)
            G_mean = np.concatenate([-G_mean, [0], G_mean])
            G_std = np.concatenate([G_std, [0], G_std])

        elif self.mode == 5:
            G_unique = np.unique(G_mean)
            G_std_map = {val: std for val, std in zip(G_mean, G_std)}

            G1, G2 = np.meshgrid(G_unique, G_unique)
            std1 = np.vectorize(G_std_map.get)(G1)
            std2 = np.vectorize(G_std_map.get)(G2)

            # 표현 가능한 조합 생성
            single_pos = G_unique
            single_neg = -G_unique
            std_single = np.vectorize(G_std_map.get)(G_unique)

            sum_vals = (G1 + G2).flatten()
            std_sum = np.sqrt(std1**2 + std2**2).flatten()

            diff_vals = (G1 - G2).flatten()
            std_diff = np.sqrt(std1**2 + std2**2).flatten()

            neg_sum_vals = (-G1 - G2).flatten()
            std_neg_sum = np.sqrt(std1**2 + std2**2).flatten()

            # 최종 후보 합치기
            G_mean_combined = np.concatenate([
                single_pos,
                single_neg,
                sum_vals,
                diff_vals,
                neg_sum_vals
            ])
            G_std_combined = np.concatenate([
                std_single,
                std_single,
                std_sum,
                std_diff,
                std_neg_sum
            ])

            # 중복 제거
            G_mean_final, indices = np.unique(G_mean_combined, return_index=True)
            G_std_final = G_std_combined[indices]

            # 클리핑은 최대 전도도 * 2 기준으로 설정
            G_max = np.max(np.abs(G_unique))
            weight_bound = 2 * G_max
            mask = (G_mean_final >= -weight_bound) & (G_mean_final <= weight_bound)

            self.G = G
            self.G_mean = G_mean_final[mask]
            self.G_std = G_std_final[mask]
            self.min_weight = -weight_bound
            self.max_weight = weight_bound
            return
        
        else:
            raise ValueError("Invalid mode")'''

        self.G = G
        self.G_mean = G_mean
        self.G_std = G_std
        self.min_weight = np.min(G_mean - G_std)
        self.max_weight = np.max(G_mean + G_std)
        # print(f"[G_mean] min: {np.min(G_mean):.6f}, max: {np.max(G_mean):.6f}, count: {len(G_mean)}")
        # print(f"[G_std]  min: {np.min(G_std):.6f}, max: {np.max(G_std):.6f}, count: {len(G_std)}")


    # 가중치 범위 제한 함수(keras는 class로만 clip 가능)
    def get_constraint(self):
        if self.use_clip:
            class Clip(tf.keras.constraints.Constraint):
                def __init__(inner_self):
                    inner_self.min_value = self.min_weight
                    inner_self.max_value = self.max_weight

                def __call__(inner_self, w):
                    return tf.clip_by_value(w, inner_self.min_value, inner_self.max_value)
                
                def get_config(inner_self):
                    return {"min_value": inner_self.min_value, "max_value": inner_self.max_value}
            return Clip()
        return None


    ''' # PTQ 양자화 함수 - 훈련 후 weight를 G_mean에 mapping, noise 추가
    def quantize(self, weights):
        q = np.copy(weights)
        for i in range(weights.shape[0]):
            idx = np.argmin(np.abs(self.G_mean - weights[i]))
            noise = np.random.normal(loc=0.0, scale=self.G_std[idx] * self.noise_scaling_factor)
            q[i] = self.G_mean[idx] + noise
        return q

    # PTQ 양자화 적용 함수
    def quantize_model(self, model):
        for layer in model.layers:
            if isinstance(layer, Dense):
                weights, biases = layer.get_weights()
                quant_weights = self.quantize(weights.flatten()).reshape(weights.shape)
                layer.set_weights([quant_weights, biases])
    
    # 양자화 함수
    def quantization(self, model):
        for layer in model.layers:
            if isinstance(layer, Dense):
                weights, biases = layer.get_weights()
                q = np.copy(weights)
                for i in range(q.shape[0]):
                    for j in range(q.shape[1]):
                        idx = np.argmin(np.abs(self.G_mean - q[i, j]))
                        noise = np.random.normal(loc=0.0, scale=self.G_std[idx] * self.noise_scaling_factor)
                        q[i, j] = self.G_mean[idx] + noise
                layer.set_weights([q, biases])'''


    # 1D 양자화
    def quantize_scalar(self, w):
        idx = np.argmin(np.abs(self.G_mean - w))
        noise = np.random.normal(loc=0.0, scale=self.G_std[idx] * self.noise_scaling_factor)
        return self.G_mean[idx] + noise
    

    # 2D 양자화
    def quantize_vector(self, weights_1d):
        return np.array([self.quantize_scalar(w) for w in weights_1d])


    # PTQ 적용 함수
    def quantization(self, model):
        for layer in model.layers:
            if isinstance(layer, Dense):
                weights, biases = layer.get_weights()
                q = np.copy(weights)
                for i in range(q.shape[0]):
                    for j in range(q.shape[1]):
                        q[i, j] = self.quantize_scalar(q[i,j])
                layer.set_weights([q, biases])


    # QAT 적용 함수
    def get_qat_callback(self, _target_model):
        """
        QAT 훈련용 콜백을 반환합니다.
        _target_model: 양자화를 적용할 실제 Keras 모델 (g_model)
        """
        class QATCallback(tf.keras.callbacks.Callback):
            def __init__(inner_self, wrapper, _target_model):
                super(QATCallback, inner_self).__init__()
                inner_self.wrapper = wrapper
                inner_self._model = _target_model

            def on_train_batch_end(inner_self, batch, logs=None):
                for layer in inner_self._model.layers:
                    if isinstance(layer, tf.keras.layers.Dense):
                        weights, biases = layer.get_weights()
                        quant_weights = inner_self.wrapper.quantize_vector(weights.flatten()).reshape(weights.shape)
                        layer.set_weights([quant_weights, biases])
                # print(f"[QAT] Batch {batch} - Quantization applied")

        return QATCallback(self, _target_model)


    # 훈련 후 가중치에 노이즈를 추가하는 함수
    def add_noise(self, model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                w, b = layer.get_weights()
                sigma = self.noise_scaling_factor * np.std(w)
                noise = np.random.normal(loc=0.0, scale=sigma, size=w.shape)
                layer.set_weights([w + noise, b])


    # 모델을 생성하는 함수
    def create_model(self):
        constraint = self.get_constraint()
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)

        model = Sequential([
            Input(shape=(28, 28)),
            Flatten(),
            Dense(128, activation="relu", kernel_initializer=initializer, kernel_constraint=constraint),
            Dense(64, activation="relu", kernel_initializer=initializer, kernel_constraint=constraint),
            Dense(10, activation="softmax", kernel_initializer=initializer, kernel_constraint=constraint)
        ])
        return model


# MNIST 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# 모델 생성
create_model = Model(**config)
g_model = create_model.create_model()
g_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# 모델 학습
qat_cb = create_model.get_qat_callback(g_model)
callbacks = []
if config.get("use_qat"):
    callbacks=[qat_cb]

g_model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)


# QAT가 적용됐는지 확인하는 함수
'''
아래와 같은 출력으로 qat가 적용되었는지 확인할 수 있다.
[Layer 1] Mismatched Weights: 0 / 100352
[Layer 2] Mismatched Weights: 0 / 8192
[Layer 3] Mismatched Weights: 0 / 640
'''
def check_out_of_range_weights(model, G_mean):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            weights = layer.get_weights()[0].flatten()
            mismatched = [w for w in weights if not np.any(np.isclose(w, G_mean, rtol=0, atol=1e-6))]
            print(f"[Layer {i}] Mismatched Weights: {len(mismatched)} / {len(weights)}")

check_out_of_range_weights(g_model, create_model.G_mean)

#%% 평가
# 양자화 모델 생성
create_model = Model(**config)
qnt_g_model = create_model.create_model()
# qnt_g_model = tf.keras.models.clone_model(g_model)
qnt_g_model.set_weights(g_model.get_weights())
if config.get("use_ptq"):
    create_model.quantization(qnt_g_model)
qnt_g_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# 훈련 후 가중치에 노이즈를 추가한 모델 생성
create_model = Model(**config)

n_g_model = create_model.create_model()
# n_g_model = tf.keras.models.clone_model(g_model)
n_g_model.set_weights(g_model.get_weights())
create_model.add_noise(n_g_model)
n_g_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

n_qnt_g_model = create_model.create_model()
# n_qnt_g_model = tf.keras.models.clone_model(qnt_g_model)
n_qnt_g_model.set_weights(g_model.get_weights())
create_model.add_noise(n_qnt_g_model)
n_qnt_g_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# weight 범위 출력 함수
def print_weight_statistics(model, name):
    print(f"\n Weight Statistics for {name}:")
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            weights = layer.get_weights()[0]
            print(f"  Layer {i}: Min Weight = {np.min(weights):.6f}, Max Weight = {np.max(weights):.6f}")


# 최종 평가
loss_g, acc_g = g_model.evaluate(x_test, y_test, verbose=0)
loss_q, acc_q = qnt_g_model.evaluate(x_test, y_test, verbose=0)
loss_ng, acc_ng = n_g_model.evaluate(x_test, y_test, verbose=0)
loss_nq, acc_nq = n_qnt_g_model.evaluate(x_test, y_test, verbose=0)


print(f"Final Evaluation Results:")
print(f"[Conductance Model]   Test Loss: {loss_g:.4f}  |  Test Accuracy: {acc_g:.4f}")
print(f"[Quantized Conductance Model] Test Loss: {loss_q:.4f}  |  Test Accuracy: {acc_q:.4f}")
print(f"[Noisy Conductance Model]   Test Loss: {loss_ng:.4f}  |  Test Accuracy: {acc_ng:.4f}")
print(f"[Noisy Quantized Conductance Model] Test Loss: {loss_nq:.4f}  |  Test Accuracy: {acc_nq:.4f}")
print_weight_statistics(g_model, "Conductance Model")
print_weight_statistics(qnt_g_model, "Quantized Conductance Model")
print_weight_statistics(n_g_model, "Noisy Conductance Model")
print_weight_statistics(n_qnt_g_model, "Noisy Quantized Model")



# 모델의 가중치, 성능 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"/Users/ch/python/quantization/results/qnt_model_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)
# g_model.save_weights(os.path.join(save_dir, "g_weights.h5"))
# qnt_g_model.save_weights(os.path.join(save_dir, "qnt_g_weights.h5"))
with open(os.path.join(save_dir, "g_metrics.json"), "w") as f:
    json.dump({"loss": float(loss_g), "accuracy": float(acc_g)}, f, indent=4)
with open(os.path.join(save_dir, "qnt_g_metrics.json"), "w") as f:
    json.dump({"loss": float(loss_q), "accuracy": float(acc_q)}, f, indent=4)
with open(os.path.join(save_dir, "n_g_metrics.json"), "w") as f:
    json.dump({"loss": float(loss_ng), "accuracy": float(acc_ng)}, f, indent=4)
with open(os.path.join(save_dir, "n_qnt_g_metrics.json"), "w") as f:
    json.dump({"loss": float(loss_nq), "accuracy": float(acc_nq)}, f, indent=4)


# 가중치 히스토그램 출력 함수
def plot_weight_hist(model, title="weight distribution", bins=50):
    w_all = np.concatenate([
        layer.get_weights()[0].flatten()
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ])
    plt.figure(figsize=(6,4))
    plt.hist(w_all, bins=bins)
    plt.title(title)
    plt.xlabel("weight value")
    plt.ylabel("count")
    plt.show(block=False)

plot_weight_hist(g_model, "Conductance(QAT) model weights")
plot_weight_hist(qnt_g_model, "Quantized(PTQ) model weights")
plot_weight_hist(n_g_model, "Noisy Conductance(QAT) model weights")
plot_weight_hist(n_qnt_g_model, "Noisy Quantized(PTQ) model weights")

