import axios from 'axios';

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const TOKEN_KEY = 'kbagent_token';

export const getToken = (): string | null => localStorage.getItem(TOKEN_KEY);
export const setToken = (token: string): void => {
  localStorage.setItem(TOKEN_KEY, token);
};

// 首次进入即签发匿名身份（服务端签发 user_id，客户端不自填），
// 否则"我的偏好"在用户聊天前访问会 401。
export const ensureToken = async (): Promise<void> => {
  if (getToken()) return;
  try {
    const res = await axios.get(`${API_BASE_URL}/api/auth/anonymous`);
    if (res.data?.token) setToken(res.data.token);
  } catch (e) {
    console.warn('匿名身份获取失败', e);
  }
};

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
client.interceptors.request.use(
  (config) => {
    // 携带服务端签发的匿名身份 token（user_id 由服务端决定，客户端不再自填）
    const token = getToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
client.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    if (error.response?.status === 500) {
      console.error('服务器错误:', error.response.data);
    }
    return Promise.reject(error);
  }
);

export default client;
