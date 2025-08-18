FROM python:3.10

ENV AIRFLOW_HOME=/usr/local/airflow
ENV AIRFLOW__CORE__AUTH_MANAGER=airflow.providers.fab.auth_manager.fab_auth_manager.FabAuthManager

RUN apt-get update && \
    apt-get install -y gcc libc-dev libssl-dev vim && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel && \
    pip install apache-airflow && \
    pip install apache-airflow-providers-fab

RUN mkdir -p $AIRFLOW_HOME
WORKDIR $AIRFLOW_HOME
RUN airflow db migrate

COPY my_dag.py $AIRFLOW_HOME/dags/

EXPOSE 8080

CMD airflow api-server  --port 8080 --host 0.0.0.0  & airflow scheduler
