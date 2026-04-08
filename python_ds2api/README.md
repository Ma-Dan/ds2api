# DS2API Python Version

将 DeepSeek Web 对话能力转换为 OpenAI 兼容 API 的 Python 实现。

## 功能特性

- **OpenAI 兼容接口**: 支持 `/v1/chat/completions`、`/v1/models`、`/v1/embeddings`
- **多账号管理**: 自动登录、Token 刷新、并发控制
- **模型别名**: 支持将 GPT-4 等模型名映射到 DeepSeek 模型
- **流式响应**: 完整支持 SSE 流式输出
- **PoW 算法**: 纯 Python 实现 DeepSeekHashV1

## 支持的模型

| 模型 | 说明 |
| --- | --- |
| `deepseek-chat` | 普通对话 |
| `deepseek-reasoner` | 深度思考模式 |
| `deepseek-chat-search` | 联网搜索模式 |
| `deepseek-reasoner-search` | 深度思考 + 联网搜索 |

## 快速开始

### 1. 安装依赖

```bash
cd python_ds2api
pip install -r requirements.txt
```

### 2. 配置

```bash
cp config.example.json config.json
# 编辑 config.json，填入你的 DeepSeek 账号和密码
```

配置示例：

```json
{
  "keys": ["your-api-key"],
  "accounts": [
    {
      "email": "your-email@example.com",
      "password": "your-password"
    }
  ],
  "model_aliases": {
    "gpt-4o": "deepseek-chat",
    "o1": "deepseek-reasoner"
  }
}
```

### 3. 启动服务

```bash
python main.py
```

或使用 uvicorn：

```bash
uvicorn main:app --host 0.0.0.0 --port 5001
```

### 4. 调用 API

```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## 环境变量

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `PORT` | 服务端口 | `5001` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `DS2API_ADMIN_KEY` | Admin 密钥 | `admin` |
| `DS2API_JWT_SECRET` | JWT 签名密钥 | 同 ADMIN_KEY |
| `DS2API_CONFIG_PATH` | 配置文件路径 | `config.json` |
| `DS2API_CONFIG_JSON` | 配置 JSON (Base64 或原始) | - |

## 使用 Python SDK 调用

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:5001/v1"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## 流式输出

```python
stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "讲个笑话"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## 深度思考模式

```python
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "解释相对论"}]
)

# 获取思考过程
if hasattr(response.choices[0].message, 'reasoning_content'):
    print("思考过程:", response.choices[0].message.reasoning_content)

print("回答:", response.choices[0].message.content)
```

## API 端点

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `/` | GET | API 信息 |
| `/healthz` | GET | 健康检查 |
| `/readyz` | GET | 就绪检查 |
| `/v1/models` | GET | 模型列表 |
| `/v1/models/{id}` | GET | 模型详情 |
| `/v1/chat/completions` | POST | 对话补全 |
| `/v1/embeddings` | POST | 文本嵌入 |

## cURL 调用示例

### 1. 健康检查

```bash
# 健康检查
curl http://localhost:5001/healthz

# 就绪检查
curl http://localhost:5001/readyz
```

### 2. 获取模型列表

```bash
curl http://localhost:5001/v1/models
```

### 3. 非流式对话

```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user", "content": "你好，请介绍一下自己"}
    ],
    "stream": false
  }'
```

### 4. 流式对话

```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "user", "content": "讲一个简短的笑话"}
    ],
    "stream": true
  }'
```

### 5. 深度思考模式 (DeepSeek Reasoner)

```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "deepseek-reasoner",
    "messages": [
      {"role": "user", "content": "解释一下相对论的基本原理"}
    ],
    "stream": false
  }'
```

### 6. 使用模型别名

```bash
# gpt-4o 会自动映射到 deepseek-chat
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### 7. 获取文本嵌入

```bash
curl http://localhost:5001/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "text-embedding-ada-002",
    "input": "这是一段需要获取嵌入向量的文本"
  }'
```

### 8. 使用 Direct Token 模式

如果你有 DeepSeek 的直接 Token，可以跳过配置直接使用：

```bash
# 将 DeepSeek Token 直接作为 API Key 传入
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_DEEPSEEK_TOKEN" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'
```

### 9. 指定特定账号

```bash
# 使用 X-Ds2-Target-Account 头指定使用某个托管账号
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -H "X-Ds2-Target-Account: your-email@example.com" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'
```

### 10. 带工具调用的对话

```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "获取指定城市的天气信息",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string",
                "description": "城市名称"
              }
            },
            "required": ["city"]
          }
        }
      }
    ]
  }'
```

## 与 Go 版本的差异

1. **运行时**: 使用 Python + FastAPI + uvicorn 代替 Go + chi
2. **HTTP 客户端**: 使用 httpx 代替 Go 的 net/http
3. **PoW 实现**: 纯 Python 实现 Keccak-256 算法
4. **并发控制**: 使用 asyncio 锁机制代替 Go 的 sync 包

## 注意事项

- 本项目仅供学习研究使用
- 请勿用于违反 DeepSeek 服务条款的场景
- Token 仅在内存中存储，服务重启后需要重新登录

## 许可证

本项目遵循原仓库的 LICENSE。
