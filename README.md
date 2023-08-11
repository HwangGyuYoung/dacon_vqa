# dacon_vqa
### [월간 데이콘 이미지 기반 질의 응답 AI 경진대회](https://dacon.io/competitions/official/236118/overview/description)
Private 1위, Accuracy 0.67214

프로젝트 디렉토리 구조
```
│dacon_vqa/
├── beit3/
│   ├── ...
│
├── models/
│   ├── beit3.spm
│   ├── beit3_base_indomain_patch16_224.zip
│   ├── beit3_large_indomain_patch16_224.zip
│
├── image/
│   ├── train/
│   │    ├── ...
│   ├── test/
│       ├── ...
│
├── train.csv
├── test.csv
│
├── make_answer2label.py
├── answer2label.txt
├── dataset.py
├── distributed_train.py
├── train.py
├── inference.py
```

beit3 폴더 안의 코드와 models 폴더 안의 모델은 아래 beit3 공식 github page에서 다운받았으며, zip파일은 압축 해제하지 않고 그대로 사용하시면 됩니다. \
https://github.com/microsoft/unilm/tree/master/beit3

또한, 경진대회에서 제공한 데이터 파일인 *open.zip*은 압축 해제하여 위와 같이 구성하면 됩니다. 

_answer2label.txt_ 파일은 *make_answer2label.py*를 실행하면 생성되며, *train.csv*를 기반으로 answer의 빈도가 높은 순서로 3129개를 뽑습니다. 모델은 최종적으로 이 3129개의 answer 중 1개를 예측하는 classification 문제를 풀게 됩니다.

학습 코드는 *distributed_train.py*와 *train.py*가 있으며 multi-gpu 환경에서 코드를 실행할 수 있으면 `DistributedDataParallel`을 구현한 코드인 *distributed_train.py*를 실행하시고, single-gpu 환경이라면 *train.py*를 실행하시면 됩니다.
multi-gpu 환경에서는 gpu 개수에 따라 _distributed_train.py_ 코드에서 `DistributedArgs` class의 `world_size`와 `gpu` 파라미터를 수정하시면 됩니다. \
또한, distributed_train.py 코드에서는 Automatic Mixed Precision 기법을 적용하여 학습 속도를 더욱 높였습니다. 최종적으로, train.py(single gpu & no amp)와 distributed_train.py(4 gpu & amp)의 학습 속도 차이는 7배 정도가 발생하였습니다. 

학습이 완료된 모델은 models 폴더에 저장되며, _inference.py_ 코드는 저장된 모델을 불러와 test 데이터를 추론하고 _submission.csv_ 파일을 생성합니다.

리더보드 상 1위를 기록한 모델은 train.py 코드 결과물이며 `CFG = {'IMG_SIZE': 480, 'EPOCHS': 10, 'LEARNING_RATE': 1e-5, 'BATCH_SIZE': 16, 'SEED': 41}`로 설정하여 7epoch이 끝나고 저장된 모델입니다.