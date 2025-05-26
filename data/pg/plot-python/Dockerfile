
FROM python:3.13

#RUN apk add build-base clang
RUN apt-get update && apt-get install -y build-essential

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./plot.py", "./result/pg-uni-222-closeness.out", "./result/out.pdf", "Payment Graph" ]

