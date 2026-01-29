# 02. LLM과 MCP 연동(실무 예시)

## 이 챕터에서 배우는 것

- FastMCP 기반 Tool 서버를 **기동하는 방법**
- 단일/다중 MCP 서버에 **연결하는 방법**
- LangGraph/LangChain과 **연동하는 방법**
- 실무에서 필요한 **전송 방식/헤더/운영 포인트**

---

## 1. 언제 이 문서를 참고해야 하나

이 문서는 **“MCP를 실제로 붙여서 돌려야 할 때”** 참고합니다. 예를 들어 다음 상황에서 필요합니다.

- Tool 서버를 띄우고 **로컬에서 바로 동작 확인**할 때
- 여러 MCP 서버를 **동시에 연결**해야 할 때
- LLM이 MCP Tool을 **실제로 호출**해야 할 때
- 인증/추적 헤더를 넣어 **운영 환경에 적용**해야 할 때

즉, 개념이 아니라 **실제 연결과 실행을 위한 문서**입니다.

---

## 2. FastMCP 서버 최소 예제(기본: stdio)

아래 예시는 FastMCP 기반으로 **Tool을 등록하고 서버를 기동**하는 최소 흐름입니다.  
공식 Quickstart는 로컬 실행에 적합한 `stdio` 전송을 사용합니다.

`stdio`는 **표준 입력/출력으로 프로세스를 연결**하는 방식입니다.  
로컬 개발이나 단일 머신 내 테스트에서 가장 단순하고 안정적이며, 별도 네트워크 설정이 필요 없습니다.

```python
"""
목적: FastMCP 기반 Tool 서버 예시를 제공한다.
설명: Tool을 등록하고 stdio로 실행한다.
디자인 패턴: 커맨드
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """두 수를 더한다."""

    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """두 수를 곱한다."""

    return a * b


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

기동 명령(uv 기준)

```bash
uv run --with mcp math_server.py
```

---

## 3. 단일 서버 연결(클라이언트 세션 + load_mcp_tools)

단일 서버는 `ClientSession`과 `load_mcp_tools` 조합이 가장 직관적입니다.

`load_mcp_tools`는 MCP가 제공하는 Tool 목록을 **LangChain Tool 객체로 변환**해 줍니다.  
따라서 단일 서버 환경에서는 “연결 → 변환 → 에이전트 연결” 흐름이 간단해집니다.

```python
"""
목적: 단일 MCP 서버에 연결해 Tool을 로드한다.
설명: stdio_client + ClientSession + load_mcp_tools를 사용한다.
디자인 패턴: 어댑터
"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

server_params = StdioServerParameters(
    command="python",
    # 절대 경로로 변경
    args=["/path/to/math_server.py"],
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_agent("openai:gpt-4.1", tools)
            result = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            return result
```

---

## 4. 다중 서버 연결(MultiServerMCPClient)

여러 MCP 서버를 동시에 연결해야 한다면 `MultiServerMCPClient`를 사용합니다.

이 방식은 여러 서버에서 Tool을 모아 **하나의 Tool 리스트로 합치는 데** 유리합니다.  
즉, Agent 입장에서는 여러 MCP 서버가 있어도 **하나의 Tool 묶음**처럼 사용할 수 있습니다.

```python
"""
목적: 다중 MCP 서버에서 Tool을 로드한다.
설명: stdio + http 혼합 연결을 지원한다.
디자인 패턴: 어댑터
"""

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            "args": ["/path/to/math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            "url": "http://localhost:8000/mcp",
            "transport": "http",
        },
    }
)

async def run():
    tools = await client.get_tools()
    agent = create_agent("openai:gpt-4.1", tools)
    r1 = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
    r2 = await agent.ainvoke({"messages": "what is the weather in nyc?"})
    return r1, r2
```

---

## 5. Streamable HTTP 전송

Streamable HTTP는 **HTTP 기반이지만 응답을 스트리밍으로 받는 전송 방식**입니다.  
요청-응답을 한 번에 끝내지 않고 **연결을 유지하며 데이터를 순차적으로 전달**합니다.  
실무에서는 표준 HTTP 환경을 유지하면서도 **부분 결과를 빨리 받거나, 긴 응답을 안정적으로 처리**할 때 유리합니다.

MCP는 `streamable HTTP` 전송을 지원하며, `streamablehttp_client`로 연결할 수 있습니다.

```python
"""
목적: streamable HTTP로 MCP 서버에 연결한다.
설명: streamablehttp_client + ClientSession을 사용한다.
"""

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

async def run():
    async with streamablehttp_client("http://localhost:3000/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_agent("openai:gpt-4.1", tools)
            return await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
```

---

## 6. 런타임 헤더 전달

`MultiServerMCPClient`는 `http`/`streamable_http` 전송에서 **런타임 헤더**를 지원합니다.  
런타임 헤더는 **요청마다 붙는 인증/추적 정보**를 의미합니다.

예시로는 다음이 있습니다.

- `Authorization`: 인증 토큰 전달
- `X-Trace-Id`: 호출 추적용 식별자
- `X-Request-Id`: 요청 단위 로깅

즉, 운영 환경에서 **보안과 추적성**을 확보하기 위해 헤더를 활용합니다.

```python
"""
목적: MCP 연결에 인증/추적 헤더를 전달한다.
설명: headers 필드를 사용한다.
"""

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
            "headers": {
                "Authorization": "Bearer YOUR_TOKEN",
                "X-Trace-Id": "trace-123",
            },
        }
    }
)

async def run():
    tools = await client.get_tools()
    agent = create_agent("openai:gpt-4.1", tools)
    return await agent.ainvoke({"messages": "what is the weather in nyc?"})
```

---

## 7. 실행 단계 요약(실무)

1. **FastMCP 서버 기동**: Tool을 등록하고 엔드포인트를 노출한다.
2. **MCP Tool 목록 수신**: Agent가 MCP에서 Tool 카드를 가져온다.
3. **에이전트 생성**: LangGraph/LangChain 에이전트에 Tool을 연결한다.
4. **실행/응답**: Tool 호출 → 결과 수집 → LLM 응답 생성.

---

## 8. 운영 포인트

- 로컬 개발은 `stdio`, 서비스 간 통신은 `http/streamable_http`가 일반적이다.
- 호출 실패 시에는 **재시도/폴백 정책**을 MCP에서 강제하는 것이 안전하다.
- 모델 설정은 사용 환경에 맞게 변경한다.

운영 관점에서 중요한 점은 **전송 방식과 에러 정책이 일관되게 유지되는가**입니다.  
특히 다중 서버 환경에서는 헤더/레이트 리밋/재시도 정책을 중앙에서 통제해야 운영 리스크가 줄어듭니다.

---

## 9. 체크리스트

- MCP 서버가 정상 기동되고 엔드포인트가 노출되는가?
- Tool 등록이 데코레이터 기반으로 표준화되어 있는가?
- LLM 에이전트가 MCP Tool을 정상적으로 호출하는가?
- 실패 시 재시도/폴백 정책이 적용되는가?
