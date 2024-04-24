# POC LangChain

## Running

Create a file `.env` with `OPENAI_API_KEY` env:

```
OPENAI_API_KEY=<my-key>
```

Once this is done, run the command below to isolate Python and our application's dependencies:

```shell
python -m venv venv
```

To activate run:

```shell
source venv/bin/activate
```

To install application's dependencies:

```shell
pip install -r requirements.txt
```

To run script files:

```shell
python <filename>.py
```