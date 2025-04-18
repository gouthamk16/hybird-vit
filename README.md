Train without mixed precision:

```bash
python -W ignore train.py --batch_size 32 --epochs 50 --use_arcface
```

Train with mixed precision

```bash
python -W ignore train_mixedPrecision.py --batch_size 64 --use_arcface --img_size 224
```