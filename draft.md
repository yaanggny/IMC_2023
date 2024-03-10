## structure
```markdown
third_party/
src/
    detector/
    matcher/
    ransac/
    base_predictor.py
        __init__(config.json | dict, pretrained_weights='file')
        load_image_pair
        predict_pair_file
        predict_pair_img
        preprocess
        postprocess

    utils/
    pipeline.py        
scripts/
config.json

env.yaml
    pytorch1.13.0 + cuda 11.6
    pytorch2.2.1 + cuda12.1
requirements.txt
    kornia kornia_moons
    opencv-python

README.md
```

## 竞赛用config
```json
{
    // input, output

    "matchers_twostep": [
        {
            "enable": true,
            "detect": {
                "model": "SuperPoint",
                "config": "m/SuperPoint.json"
            },
            "match": {
                "model": "SuperGlue",
                "config": "m/SuperGlue.json"
            }
        },
        {
            "enable": true,
            "detect": {},
            "match": {}
        }
    ],
    "macthers_onestep": [
        {
            "enable": true,
            "model": "loftr",
            "config": "m/loftr.json"
        }
    ],
    "ransac": {   // model.load_config(xx.json)
        "model": "std",
        "config": "m/ransac.json"
    }    
}
```