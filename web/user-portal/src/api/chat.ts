import client, { API_BASE_URL, getToken, setToken } from './client';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

export interface ChatRequest {
  user_id?: string; // 已废弃：身份由服务端签发 token 决定
  message: string;
  session_id?: string;
  context?: Record<string, any>;
}

export interface ChatResponse {
  success: boolean;
  reply: string;
  intent?: string;
  tool_used?: string;
  context_count?: number;
  session_id: string;
  token?: string;
  timestamp: string;
}

export interface SessionInfo {
  session_id: string;
  user_id: string;
  history_length: number;
  created_at: string;
  last_active: string;
}

export const chatApi = {
  // 发送消息
  sendMessage: (data: ChatRequest): Promise<ChatResponse> => {
    return client.post('/chat', data);
  },

  // 获取会话历史
  getSessionHistory: (sessionId: string): Promise<{ session_id: string; history: any[]; turn_count: number }> => {
    return client.get(`/sessions/${sessionId}/history`);
  },

  // 获取活跃会话列表
  getSessions: (): Promise<SessionInfo[]> => {
    return client.get('/sessions');
  },

  // 删除会话
  deleteSession: (sessionId: string): Promise<{ success: boolean; message: string }> => {
    return client.delete(`/sessions/${sessionId}`);
  },

  // 流式发送消息（SSE）：token 逐段回调，done 携带完整回复
  sendMessageStream: (
    data: ChatRequest,
    handlers: {
      onToken: (content: string) => void;
      onDone: (payload: any) => void;
      onError: (message: string) => void;
    },
  ): void => {
    const token = getToken();
    fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(data),
    })
      .then(async (resp) => {
        if (!resp.ok || !resp.body) {
          handlers.onError(`请求失败（${resp.status}）`);
          return;
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split('\n\n');
          buffer = events.pop() || '';
          for (const ev of events) {
            const line = ev.split('\n').find((l) => l.startsWith('data: '));
            if (!line) continue;
            const payload = JSON.parse(line.slice(6));
            if (payload.type === 'token') {
              handlers.onToken(payload.content);
            } else if (payload.type === 'done') {
              if (payload.token) setToken(payload.token);
              handlers.onDone(payload);
            } else if (payload.type === 'error') {
              handlers.onError(payload.message);
            }
          }
        }
        // 流结束时可能残留最后一个未以 \n\n 结尾的事件（健壮性兜底）
        if (buffer.trim()) {
          const line = buffer.split('\n').find((l) => l.startsWith('data: '));
          if (line) {
            const payload = JSON.parse(line.slice(6));
            if (payload.type === 'token') {
              handlers.onToken(payload.content);
            } else if (payload.type === 'done') {
              if (payload.token) setToken(payload.token);
              handlers.onDone(payload);
            } else if (payload.type === 'error') {
              handlers.onError(payload.message);
            }
          }
        }
      })
      .catch((e) => handlers.onError(String(e)));
  },
};
