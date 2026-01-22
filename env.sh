uv venv --python 3.10 --seed
git clone git@github.com:haotian-liu/LLaVA.git

source .venv/bin/activate
# install lmms_eval without building dependencies
UV_HTTP_TIMEOUT=3600
uv pip install -e ".[all]"
# uv pip install loguru wandb nltk spacy
# uv pip install httpx==0.23.3

# install LLaVA without building dependencies
cd LLaVA
uv pip install -e .


cd ..

# install all the requirements that require for reproduce llava results
# uv pip install -r miscs/llava_repr_requirements.txt

# Run and exactly reproduce llava_v1.5 results!
# mme as an example
# accelerate launch --num_processes=8 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-7b,device_map=auto"   --tasks mme  --batch_size 1 --log_samples --log_samples_suffix reproduce --output_path ./logs/ --verbosity=DEBUG
