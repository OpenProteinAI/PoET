create_lock:
	cd environments/poet && mamba run -n base conda-lock -f environment.yml -p linux-64 --lockfile conda-lock.yml

create_stable_env:
	mamba run -n base conda-lock install -n poet environments/poet/conda-lock.yml
	mamba run -n poet mamba env update -n poet -f environments/poet/environment.dev.yml
	DS_BUILD_CPU_ADAM=1 BUILD_UTILS=1 mamba run -n poet pip install -r environments/poet/requirements_stable.txt

precommit:
	mamba run -n poet pre-commit install

create_conda_env: create_stable_env precommit

update_conda_env: create_lock create_conda_env

download_model:
	wget -c https://zenodo.org/records/10061322/files/poet.ckpt?download=1 -O data/poet.ckpt
	wget -c https://zenodo.org/records/10061322/files/LICENSE?download=1 -O data/LICENSE
