import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd


class Model:
    def __init__(self, filepath, mode=1, clip=True, G_scaling=True, scaling_factor=1e4, noise_scaling_factor=0):
        self.filepath = filepath
        self.mode = mode
        self.use_clip = clip
        self.use_scaling = G_scaling
        self.scaling_factor = scaling_factor
        self.noise_scaling_factor = noise_scaling_factor

        self.G = None
        self.G_mean = None
        self.G_std = None
        self.min_weight = None
        self.max_weight = None

        self._load_conductance()

    def _load_conductance(self):
        df = pd.read_csv(self.filepath, sep='\t', header=None)
        G = df.to_numpy()
        if self.use_scaling:
            G = G * self.scaling_factor

        G_stats = np.array([[np.mean(row[1:]), np.std(row[1:])] for row in G])
        G_mean = G_stats[:, 0]
        G_std = G_stats[:, 1]

        # 모드 선택
        if self.mode == 1:
            pass

        elif self.mode == 2:

            G_mean -= np.min(G_mean)

        elif self.mode == 3:
            G_mean = np.concatenate([-G_mean, [0], G_mean])
            G_std = np.concatenate([G_std, [0], G_std])

        elif self.mode == 4:
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
            raise ValueError("Invalid mode")

        self.G = G
        self.G_mean = G_mean
        self.G_std = G_std
        self.min_weight = np.min(G_mean - G_std)
        self.max_weight = np.max(G_mean + G_std)

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

    # weight를 G_mean에 mapping하고 noise 추가
    def quantize(self, weights):
        q = np.copy(weights)
        for i in range(weights.shape[0]):
            idx = np.argmin(np.abs(self.G_mean - weights[i]))
            noise = np.random.normal(loc=0.0, scale=self.G_std[idx] * self.noise_scaling_factor)
            q[i] = self.G_mean[idx] + noise
        return q

    def quantize_model(self, model):
        for layer in model.layers:
            if isinstance(layer, Dense):
                weights, biases = layer.get_weights()
                quant_weights = self.quantize(weights.flatten()).reshape(weights.shape)
                layer.set_weights([quant_weights, biases])
    
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
    
# 
model_config = Model(
    filepath='/Users/ch/dataset/quant/Depression_voltage_arranged/V_D = -1.6 V',
    mode=3, clip=True, G_scaling=True, scaling_factor=1e4, noise_scaling_factor=0
)

# MNIST 데이터셋 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 모델 생성
trained_model = model_config.create_model()
trained_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 모델 학습
trained_model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    # callbacks=[EvaluationCallback()],
    verbose=1
)

# 모델 복사
copied_model = trained_model

# 모델 양자화
quantized_model = model_config.create_model()
quantized_model.set_weights(trained_model.get_weights())
quantized_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model_config.quantize_model(quantized_model)

# 양자화 적용 콜백 정의
'''class EvaluationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n[Epoch {epoch + 1}] Quantizing model weights...")

        for layer_t, layer_c in zip(trained_model.layers, copied_model.layers):
            if isinstance(layer_t, Dense):
                layer_c.set_weights(layer_t.get_weights())

        for layer_c, layer_q in zip(copied_model.layers, quantized_model.layers):
            if isinstance(layer_c, Dense):
                weights, biases = layer_c.get_weights()
                quant_weights = model_config.quantize(weights.flatten()).reshape(weights.shape)
                layer_q.set_weights([quant_weights, biases])

        _, acc_tr = trained_model.evaluate(x_test, y_test, verbose=0)
        _, acc_q = quantized_model.evaluate(x_test, y_test, verbose=0)
        print(f"Accuracy | Trained: {acc_tr:.4f} | Quantized: {acc_q:.4f}")
'''

# 최종 평가
loss_tr, acc_tr = trained_model.evaluate(x_test, y_test, verbose=2)
loss_q, acc_q = quantized_model.evaluate(x_test, y_test, verbose=2)

print(f"Final Evaluation Results:")
print(f"[Trained Model]   Test Loss: {loss_tr:.4f}  |  Test Accuracy: {acc_tr:.4f}")
print(f"[Quantized Model] Test Loss: {loss_q:.4f}  |  Test Accuracy: {acc_q:.4f}")

# weight 범위 출력 함수
def print_weight_statistics(model, name):
    print(f"\n Weight Statistics for {name}:")
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            weights = layer.get_weights()[0]
            print(f"  Layer {i}: Min Weight = {np.min(weights):.6f}, Max Weight = {np.max(weights):.6f}")

print_weight_statistics(trained_model, "Trained Model")
print_weight_statistics(quantized_model, "Quantized Model")
