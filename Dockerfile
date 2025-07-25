FROM continuumio/miniconda3:latest

# Create conda environment
WORKDIR /tmp
COPY environment.yml conda-linux-64.lock ./
RUN conda create --name demos-env --file conda-linux-64.lock \
	&& conda clean --all --yes \
	&& rm conda-linux-64.lock environment.yml

# Copy the code
COPY ./demos /demos
WORKDIR /demos

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "demos-env", "python", "-u", "simulate.py", "-cfg", "config.toml"]
