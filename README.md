Claro — aqui vai uma versão mais forte, limpa e profissional do `README.md`, com linguagem mais técnica, organização melhor e foco em GitHub:

````markdown
# Distributed Inference Cluster Core

Solução de infraestrutura distribuída para inferência em rede local (**LAN**), projetada para orquestrar cargas de trabalho de **LLM**, **TTS** e outros pipelines de IA com foco em eficiência, modularidade e baixa latência.

O projeto foi desenhado para operar como núcleo de um cluster distribuído, integrando:

- **Control Plane**
- **Data Plane**
- **Execution Plane**
- **Dashboard de observabilidade**
- **Autodescoberta automática de nós**

---

## Visão Geral

O sistema coordena nós orquestradores e workers em uma arquitetura distribuída, permitindo:

- Distribuição inteligente de tarefas
- Execução local ou remota
- Balanceamento entre CPU e GPU
- Fallback automático em caso de indisponibilidade de acelerador
- Monitoramento em tempo real

---

## Funcionalidades

- **Autodescoberta via UDP Beacon**  
  Os nós podem localizar automaticamente o orquestrador na mesma rede local.

- **Fila distribuída com prioridade**  
  Suporte a agendamento, deduplicação, timeout e controle de backpressure.

- **Arquitetura modular**  
  Separação clara entre coordenação, execução e monitoramento.

- **Suporte a múltiplos backends**  
  Integração com diferentes motores e bibliotecas, conforme disponibilidade do ambiente.

- **Fallback inteligente**  
  Se GPU não estiver disponível, o sistema pode operar em CPU.

- **Dashboard integrado**  
  Interface web para acompanhar status, métricas e execução.

---

## Arquitetura

```text
┌──────────────────────────────────────────────────────────────┐
│                     Control Plane                            │
│                 Orchestrator / Master Node                  │
│  - Registro de workers                                      │
│  - Fila de tarefas                                          │
│  - Distribuição de carga                                    │
│  - Monitoramento de saúde                                   │
└──────────────────────────────┬───────────────────────────────┘
                               │
               ┌───────────────┼────────────────┐
               │               │                │
        ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
        │   Worker 1  │ │   Worker 2  │ │   Worker N  │
        │   CPU / GPU │ │   CPU / GPU │ │   CPU / GPU │
        │ - Execução  │ │ - Execução  │ │ - Execução  │
        │ - Cache     │ │ - Cache     │ │ - Cache     │
        └─────────────┘ └─────────────┘ └─────────────┘

                     ┌──────────────────────┐
                     │   Dashboard Web      │
                     │  Métricas / Logs     │
                     └──────────────────────┘
````

---

## Requisitos

### Básicos

* Python 3.10 ou superior

### Opcionais

Dependendo do tipo de execução, podem ser instaladas bibliotecas adicionais:

```bash
pip install torch transformers psutil
```

Outros backends podem ser adicionados conforme o ambiente e o tipo de inferência desejada.

---

## Instalação

Clone o repositório:

```bash
git clone  https://github.com/albertogeraldolangaa-sketch/distributed-inference-cluster-core.git
cd distributed-inference-cluster-core
```

 
## Como Usar

### 1. Iniciar o Orquestrador

Em uma máquina principal da rede:

```bash
python distributed_inference_cluster_core.py --role orchestrator --host 0.0.0.0 --port 8000
```

O orquestrador ficará responsável por coordenar os workers e distribuir as tarefas.

---

### 2. Iniciar um Worker

Em outro nó da rede:

```bash
python distributed_inference_cluster_core.py --role worker --orchestrator http://<IP_DO_ORQUESTRADOR>:8000 --gpu
```

Se o nó for apenas CPU:

```bash
python distributed_inference_cluster_core.py --role worker --orchestrator http://<IP_DO_ORQUESTRADOR>:8000
```

---

### 3. Autodescoberta na LAN

Se todos os nós estiverem na mesma rede local, o sistema pode usar UDP Beacon para localizar o orquestrador automaticamente.

Basta iniciar os workers normalmente, com a escuta ativa habilitada no ambiente suportado.

---

## API

O sistema expõe endpoints para submissão e acompanhamento de tarefas.

### Exemplo de requisição

```bash
curl -X POST http://127.0.0.1:8000/submit \
  -H "Content-Type: application/json" \
  -d '{
    "kind": "llm",
    "priority": 50,
    "payload": {
      "prompt": "Explique paralelismo de tensores em uma frase.",
      "max_new_tokens": 120
    }
  }'
```

---

## Boas Práticas de Performance

* Prefira **LAN** ou rede privada para reduzir latência
* Use **GPU** apenas nos nós que realmente precisarem de aceleração
* Evite carregar múltiplos modelos pesados ao mesmo tempo em máquinas fracas
* Reduza o tamanho dos chunks e lotes quando o hardware for limitado
* Se possível, mantenha o **Control Plane** leve e o **Data Plane** separado
* Para payloads grandes, prefira transporte eficiente em vez de JSON/Base64 quando possível

---

## Casos de Uso

* Inferência distribuída de LLM
* Pipeline de TTS em baixa latência
* Cluster de laptops
* Processamento local sem dependência de cloud
* Ambientes offline ou com rede limitada

---

## Roadmap

* [ ] Suporte a streaming em tempo real
* [ ] Balanceamento dinâmico por capacidade do worker
* [ ] Failover automático
* [ ] Métricas detalhadas por nó
* [ ] Integração com dashboard web avançado
* [ ] Otimização para hardware de baixo consumo

---

## Autor

Desenvolvido por **Alberto Langa**
Soluções em infraestrutura distribuída, inteligência artificial e automação.

---

## Licença

Este projeto está sob licença **MIT**.

 
