# DeepLab

`deeplabpaddle`�Ǹ��ֵ�paddle����
`deeplabpytorch`��ԭpytorch����
`torch2paddle.py`��ģ��ת���ļ�

��Ҫ��װ��������omegaconf��pydensecrf��pydensecrf��Windows�������https://www.lfd.uci.edu/~gohlke/pythonlibs/#pydensecrf���ҵ���װ�������£�
```
pip install pydensecrf-1.0rc2-cp37-cp37m-win_amd64.whl
pip install omegaconf
```
����git�ϴ����ƣ�ģ����Ҫ��ʹ��torch2paddle.py����ת��
```
python torch2paddle.py
```

����paddle������ʹ��
```
cd deeplabpaddle
python run.py train --config-path configs/voc12.yaml --cuda
```
paddle���յ÷��ļ���`deeplabpaddle/data/scores/voc12/deeplabv2_resnet101_msc/val/scores.json`�У���VOC12�ϵ�miouΪ0.8987213899844988
pytorch���յ÷��ļ���`deeplabpytorch/data/scores/voc12/deeplabv2_resnet101_msc/val/scores.json`�У���VOC12�ϵ�miouΪ0.8987214672262349

