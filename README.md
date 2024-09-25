# AI-Challenge-2023

You can see the presentation of the solution here: [link](./presentation.pdf)

First place solution for medicine topic in AI Challenge 2023

First of all install all dependencies:

    $ pip install -r inference/requirements.txt

Edit paths in `inference/main.py`. Example:

```python
answer = predict_inference('/Users/danil/AIIJC_FINAL/DATA/train', display_time=True)

dump_pickle('/Users/danil/AIIJC_FINAL/DATASETS/last_predict_train.pickle', answer)
```

Then run:
```shell
python inference/main.py
```
