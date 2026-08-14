import client from './client';

export interface MemoryFact {
  text: string;
  updated_at?: string;
  source?: string;
}

export interface UserMemory {
  user_id: string;
  profile: {
    company?: string | null;
    contact_email?: string | null;
    preferences?: {
      extracted_facts?: MemoryFact[];
    };
  };
  running_summary: string;
  audit_md: string;
}

export const memoryApi = {
  // 查看用户记忆（画像 + 摘要 + 审计日志）
  getMemory: async (userId: string): Promise<UserMemory> => {
    return (await client.get(`/api/admin/memory/${userId}`)) as UserMemory;
  },

  // 删除单条偏好（人工纠错）
  deleteFact: async (userId: string, text: string): Promise<{ message: string }> => {
    return (await client.delete(`/api/admin/memory/${userId}/facts`, {
      data: { text },
    })) as { message: string };
  },

  // 清空用户记忆
  clearMemory: async (userId: string): Promise<{ message: string }> => {
    return (await client.delete(`/api/admin/memory/${userId}`)) as {
      message: string;
    };
  },
};
