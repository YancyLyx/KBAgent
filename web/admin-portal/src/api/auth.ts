import client from './client';

export interface AdminLoginResponse {
  token: string;
  username: string;
  expires_in: number;
}

export const authApi = {
  login: async (username: string, password: string): Promise<AdminLoginResponse> => {
    return (await client.post('/api/admin/login', { username, password })) as AdminLoginResponse;
  },
};
