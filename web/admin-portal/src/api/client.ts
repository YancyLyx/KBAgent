import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const ADMIN_TOKEN_KEY = 'smartsupport_admin_token';

export const getAdminToken = (): string | null =>
  localStorage.getItem(ADMIN_TOKEN_KEY);
export const setAdminToken = (token: string): void => {
  localStorage.setItem(ADMIN_TOKEN_KEY, token);
};
export const clearAdminToken = (): void => {
  localStorage.removeItem(ADMIN_TOKEN_KEY);
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
    // 管理接口统一携带 admin token
    const token = getAdminToken();
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
    if (error.response?.status === 401) {
      // token 失效/未登录：清 token 并触发全局重新登录
      clearAdminToken();
      window.dispatchEvent(new CustomEvent('admin:unauthorized'));
    }
    if (error.response?.status === 404) {
      console.error('API not found:', error.config?.url);
    }
    return Promise.reject(error);
  }
);

export default client;
