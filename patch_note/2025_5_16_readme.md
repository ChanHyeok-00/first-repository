# Quantizaiton with conductance of Neuromorphic Device

## parameters

### 1.mode: 전도도를 다루는 변수

	mode1 : 전도도를 그대로 사용
	G
	mode2 : 전도도의 최솟값이 0이 되도록 이동
	G -= G_min
	mode3 : 전도도의 범위를 음수로 확장
	G = (G, 0, G)
	* mode:3일 때 성능이 가장 좋음

### 2.scaling factor: 전도도의 크기를 변형하는 변수
	
	전도도값은 대략 수um수준이고, 기본 baseline모델의 가중치의 절댓값은 1이내이므로 스케일링이 필요함
	* scaling_factor:1e4 근처에서 성능이 가장 좋음

### 3.noise_scaling_factor: 노이즈의 크기를 변형하는 변수

	noise는 전도도의 표준편차를 이용해서 구현
	qat에 noise injection 기법을 사용해서 소자의 변동성을 학습할 수 있음
	* 파라미터를 잘 설정했을 때 50배의 noise에서도 90% 이상의 정확도를 가짐

### 4.use_int_levels: int_min 변수와 int_max 변수를 사용할지 결정하는 함수

	use_int_levels:True 인 경우 전도도 대신 int_min,int_max 범위의 정수를 가중치로 사용
	* scale과 zero point를 사용하지 않아서 모델의 가중치가 -1, +1의 값만을 가지고, framework_ptq 모델은 오류가 나온다
	* custom ptq, qat 모델의 경우 -1,+1의 가중치만으로도 각각 70%, 90% 이상의 성능을 가진다.

## model

### 1.Baseline_Float32: 
	
	TF를 사용해서 아무 조건 없이 만든 간단한 딥러닝 모델	 
	정확도는 대부분의 경우 98%

### 2.Framework_PTQ_int8:

	TF의 PTQ 기능을 사용해서 baseline모델을 양자화한 모델
        정확도는 대부분의 경우 90% 이상
	
### 3.Framework_QAT_*:
	
	TF의 QAT 기능을 사용해서 baseline모델을 양자화한 모델
	정확도는 대부분의 경우 98% 이상
	
### 5.Custom_PTQ_G_mean_ONLY:
	
	Baseline 모델의 가중치를 가장 가까운 스케일링 전도도 값으로 매핑
	정확도는 70~80% 

### 6.Custom_PTQ_G_mean_Noise:
	
	Baseline 모델의 가중치를 가장 가까운 스케일링 전도도 값으로 매핑
	매핑된 가중치에 noise추가
	noise = N(0,스케일된 std)
	정확도는 70~80%
	noise_scaling_factor가 커질수록 성능이 감소	

### 7.Custom_QAT_G_mean_Noise_In_Training:

	매 epoch마다 Baseline 모델의 가중치를 가장 가까운 스케일링 전도도 값으로 매핑
       	매 epoch마다 매핑된 가중치에 noise추가 
        noise = N(0,스케일된 std)
	정확도는 대략 95%
	noise_scaling_factor가 커져도(50정도) 90%대의 정확도 유지
