# Quantized Model Based on Conductance of Neuromorphic Devices

# 변수명

g_model: 학습 중 가중치의 범위를 스케일링한 전도도의 범위로 cliping
qnt_g_model: 학습이 완료된 g_model의 가중치를 가장 근접한 전도도 값으로 변환,mapping
n_* : add_noise 함수를 추가한 model
round_G_mean: 전도도 값을 양자화 시키는 인자.
        전도도 값의 유효숫자르 1개로 만든 후 scaling 하면 정수로 양자화한 것과 비슷하$
        Depression_voltage_arranged/V_D = -1.6 V에서 round_G_mean = [4,5,6,7,8,9]
        
# 2025.5.9 update

## class Model 변경사항

	1) 양자화(가중치를 전도도로 매핑) 함수 추가
		def quantize_scalar(1D)
		def quantize_vector(2D)
		def quantization(PTQ)
		def get_qat_callback(QAT)
		* config에서 round_G_mean 인자
			가중치를 유효숫자 1개로 반올림
			이를 스케일링하면 정수로 양자화 하는 효과처럼 보일거라 생각하고 실험 중
			아직 정확도가 0.25 수준 
			
	2) 훈련 후 가중치에 노이즈를 추가하는 함수 추가
		def add_noise
		사용되는 noise는 가중치의 std를 사용
		
	3) qat 적용 여부를 확인하는 함수 추가
		def check_out_of_range_weights

	4) class Model에서 round_G_mean, use_ptq, use_qat를 인자로 사용

## 출력 변경사항

	1) g_model, qnt_g_model에noise를 추가한 n_g_model, n_qnt_g_model의 성능도 출력
	
	2) 가중치 히스토그램 추가

## 결과

	1) ptq만 했을 때
		g_model, qnt_g_model의 정확도는 대략 90%		

	2) qat를 했을 때
		모델의 정확도는 대략 84% 로 감소
	
	3) 훈련 후 가중치에 noise를 추가했을 때 		
		n_g_model - 정확도가 대략 40%로 감소
		n_qnt_g_model - 정확도가 대략 50%로 감소

