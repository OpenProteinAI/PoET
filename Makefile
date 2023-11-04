create_lock:
	cd environments/$(env_name) && mamba run -n base conda-lock -f environment.yml -p linux-64 --lockfile conda-lock.yml

create_stable_env:
	mamba run -n base conda-lock install -n $(env_name) environments/$(env_name)/conda-lock.yml
	mamba run -n $(env_name) mamba env update -n $(env_name) -f environments/$(env_name)/environment.dev.yml
	DS_BUILD_CPU_ADAM=1 BUILD_UTILS=1 mamba run -n $(env_name) pip install -r environments/$(env_name)/requirements_stable.txt

precommit:
	mamba run -n $(env_name) pre-commit install

create_conda_env: create_stable_env precommit

update_conda_env: create_lock create_conda_env

download_model:
	wget -c https://zenodo.org/records/10061322/files/poet.ckpt?download=1 -O data/poet.ckpt
	wget -c https://zenodo.org/records/10061322/files/LICENSE?download=1 -O data/LICENSE
