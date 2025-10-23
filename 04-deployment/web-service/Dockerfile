FROM python:3.12-slim

RUN pip install -U pip
RUN pip install flask gunicorn scikit-learn

WORKDIR /app
COPY [ "predict.py", "lin_reg.bin", "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]

