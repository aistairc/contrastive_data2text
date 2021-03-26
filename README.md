# contrastive_data2text
Codes for ["Learning with Contrastive Examples for Data-to-text Generation"](https://www.aclweb.org/anthology/2020.coling-main.213/) at COLING2020.

## Resource
We used the same resources as the previous work: Aoki et al. (INLG 2018).
Please refer the "Resource" section in previous work's repo: [market-reporter](https://github.com/aistairc/market-reporter).
Since it consists of purchased tick data and news articles, we cannot make it publicly available.
Instead, we created [pseudo dataset](resources/pseudo-data) to show an example of the data format.
Note that this pseudo dataset is completely fictional and cannot be used for real training or evaluation.

## Requirements
- python 3.6
- pytorch 1.2.0
- torchtext 0.3.1 (This code doesn't support torchtext >= 0.4.0)
- nltk 3.4.5

## Training
- Edit config.toml to specify optional parameters.
- To do contrastive learning, set "neg_enabled = true", and select "neg_loss_type" from {"sentence", "token", "unlikelihood"}.
- Put datasets named alignment-train.json, alignment-valid.json and alignment-test.json under the directory specified by "dir_output" (you can set this in config.toml).
- A directory that contains trained model and log will be created under the "dir_output". You can optionally set "--output_subdir" by command for directory classification.
```buildoutcfg
python -m reporter --device 'cuda:0' --config config-example.toml --output_subdir <optinal subdirectory name>
```

## Prediction
- Use the same config.toml as one that used to train the model for prediction
```buildoutcfg
python -m reporter.evaluate --device 'cuda:0' --config config-example.toml --model_dir <directory that contains the model for prediction>
```

## Citation
```buildoutcfg
@inproceedings{uehara-etal-2020-learning,
    title = "Learning with Contrastive Examples for Data-to-Text Generation",
    author = "Uehara, Yui  and
      Ishigaki, Tatsuya  and
      Aoki, Kasumi  and
      Noji, Hiroshi  and
      Goshima, Keiichi  and
      Kobayashi, Ichiro  and
      Takamura, Hiroya  and
      Miyao, Yusuke",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.213",
    doi = "10.18653/v1/2020.coling-main.213",
    pages = "2352--2362"
}
```

## License
This code contains modified codes of the [market-reporter](https://github.com/aistairc/market-reporter).
This code is available under different licensing options:

+ [GNU General Public License (v3 or later)](https://www.gnu.org/licenses/gpl-3.0.en.html).
+ Commercial licenses.

Commercial licenses are appropriate for development of proprietary/commercial software where you do not want to share any source code with third parties or otherwise cannot comply with the terms of the GNU.
For details, please contact us at: kirt-contact-ml@aist.go.jp

This code uses a technique applied for patent (patent application number 2017001583).

## Copyright
©　2021 　National Institute of Advanced Industrial Science and Technology (AIST)