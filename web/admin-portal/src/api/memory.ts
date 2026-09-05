import client from './client';

export interface MemoryFact {
  text: string;
  updated_at?: string;
  source?: string;
}

export interface PrefEntry {
  id: string;
  text: string;
  updated_at?: string;
  source?: string;
  status?: string;
}

export interface UserMemory {
  user_id: string;
  profile: {
    topics?: any[];
    company?: string | null;
    contact_email?: string | null;
    preferences?: {
      extracted_facts?: MemoryFact[];
    };
  };
  running_summary: string;
  audit_md: string;
  prefs_timeline?: PrefEntry[];
  prefs_active?: PrefEntry[];
}

export const memoryApi = {
  // 查看用户记忆（画像 + 摘要 + 审计日志）
  getMemory: async (userId: string): Promise<UserMemory> => {
    return (await client.get(`/api/admin/memory/${userId}`)) as UserMemory;
  },

  // 删除单条偏好（人工纠错；自动提取画像）
  deleteFact: async (userId: string, text: string): Promise<{ message: string }> => {
    return (await client.delete(`/api/admin/memory/${userId}/facts`, {
      data: { text },
    })) as { message: string };
  },

  // 删除单条手动偏好（前端「我的偏好」添加，按 pref_id 软删除）
  deleteManualPref: async (userId: string, prefId: string): Promise<{ message: string }> => {
    return (await client.delete(
      `/api/admin/memory/${userId}/prefs/${prefId}`,
    )) as { message: string };
  },

  // 清空用户记忆
  clearMemory: async (userId: string): Promise<{ message: string }> => {
    return (await client.delete(`/api/admin/memory/${userId}`)) as {
      message: string;
    };
  },
};
