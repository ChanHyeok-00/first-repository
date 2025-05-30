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

### 4.num_of_G: 실험 데이터의 전도도 값 100개 중 사용할 값의 수를 정하는 변
	
	Quantization은 int16, int8, int4 등 사용하는 가중치 숫자의 수를 제한하므로 이를 구현하기 위한 변수
	PTQ의 경우 num_of_G가 100, 64, 32, 16, 8, 4, 2, 1로 변할수록 정확도가 90%대에서 60%대로 감소하지만
	QAT의 경우 90%대의 성능을 유지한다(mode 3 기준)

### 5.G_sampling_method: 총 100개의 전도도 값에서 사용할 값을 고르는 방식을 정하는 변수

	"uniform_interval": np.linspace를 사용하여 일정한 간격으로 전도도 값을 채택 
	"random": np.random을 사용하여 랜덤하게 전도도 값을 채택

### 6. apply_on_batch: QAT를epoch마다 실행할지, batch마다 실행할지 결정하는 변수

	False: 기본값, 매 epoch마다 QAT를 수행하며, 더 적은 시간, 더 적은 신뢰도를 가진다.
	True: 매 batch마다 QAT를 수행한다.
	기본적으로 Mnist data는 매 epoch마다 60,000개의 데이터를 학습하며, batch size가 64인 경우 938번의 QAT가 수행되므로
	on_train_batch_end 함수에서 몇 번의 batch마다 QAT를 수행할 지 정한다(현재는 batch % 100 == 0 일 때만 수행하도록 되어있다).

## model

### 1.Baseline_Float32: 
	
	TF를 사용해서 아무 조건 없이 만든 간단한 딥러닝 모델	 
	정확도는 대부분의 경우 98%

### 2.Custom_PTQ_G_mean_Noise:
	
	Baseline 모델의 가중치를 가장 가까운 스케일링 전도도 값으로 매핑
	매핑된 가중치에 noise추가
	noise = N(0,스케일된 std)
	정확도는 70~80%
	noise_scaling_factor가 커질수록 성능이 감소
 
### 3.Custom_PTQ_G_mean_Only:
	
	Baseline 모델의 가중치를 가장 가까운 스케일링 전도도 값으로 매핑
	정확도는 70~80% 	

### 4.Custom_QAT_G_mean_Noise_In_Training:

	매 epoch마다 Baseline 모델의 가중치를 가장 가까운 스케일링 전도도 값으로 매핑
       	매 epoch마다 매핑된 가중치에 noise추가 
        noise = N(0,스케일된 std)
	정확도는 대략 95%
	noise_scaling_factor가 커져도(50정도) 90%대의 정확도 유지

 ## step size(bit)에 대한 accuracy, loss 그래프
int8, int6, int4, int2, int1을 각각 num_of_G = 100, 32, 8, 2, 1과 mode: 3로 구현 (step size = 201, 65, 17, 5, 3)
![all_test_accuracies_vs_num_g_qats](https://github.com/user-attachments/assets/f05cbe87-5aca-470e-8225-602e9e42f545)
![all_test_losses_vs_num_g_qats](https://github.com/user-attachments/assets/30dc9488-c6bc-4961-9b73-638c4389f03f)
![noise30_all_accuracies_vs_num_g](https://github.com/user-attachments/assets/c72156d2-e33c-464b-b590-0c07afad4b51)
![noise40_all_accuracies_vs_num_g](https://github.com/user-attachments/assets/adfeb68b-8e6c-4ca5-997d-2af0014f39fc)
![image](https://github.com/user-attachments/assets/6b9e98bc-526a-4950-94c1-c0fc35a94aba)
![image](https://github.com/user-attachments/assets/151e374c-9241-4e29-a2c6-fcdced2a0ed9)

