# POC LangChain

## Install

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

## Cat Name App

To run script files:

```shell
python cat-name-app/cat-name.py
```

## Albums App

To run script files:

```shell
python albums-app/albums.py
```

Example:

![](https://cdn.discordapp.com/attachments/1044290423943876783/1234257836025053194/image.png?ex=663013af&is=662ec22f&hm=9daebc0ce6b516a5cf5e3a41befe58de8c566cda709f19954e780b868a7f4972&)

## PDF Chat

To run script files:

```shell
python -m streamlit run pdf-chat/streamlit_app.py
```

Example:

PDF
![](https://cdn.discordapp.com/attachments/1044290423943876783/1234588687174729850/image.png?ex=663147d0&is=662ff650&hm=4dec3f6ce47657a8e6d7c4527214a53b73d3c3850440ae98640ff17efa498d8f&)

Source: https://www.sbc.org.br/images/flippingbook/computacaobrasil/computa_39/pdf/CompBrasil_39_180.pdf

Chat
![](https://cdn.discordapp.com/attachments/1044290423943876783/1234629799813845032/Gravacao_de_Tela_2024-04-29_as_19.06.04_1.gif?ex=66316e1a&is=66301c9a&hm=ebc47d7989c57e6b77442e38e72a152a941ac041f885bb6f2cc511f27a77f626&)
