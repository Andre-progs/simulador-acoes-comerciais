# Simulador de Ações Comerciais

Simula o impacto de descontos e bonificações no sell-in, sell-out, receita e lucro bruto de um portfólio de dermocosméticos.
Todos os dados são fictícios e gerados pelo próprio app.

## Como rodar

1. Instale o Python (3.11 ou mais novo) em python.org, marcando "Add Python to PATH".
2. Abra esta pasta no VS Code (File > Open Folder).
3. No terminal (Ctrl+'):

```
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

No Windows, também dá para dar dois cliques em `rodar.bat`.

O app abre no navegador em http://localhost:8501 (ou na próxima porta livre, se essa estiver ocupada). Não use o botão de play do VS Code.
