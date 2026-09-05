import React, { useState, useCallback, useEffect } from 'react';
import { message as antMessage, Modal, Button, List, Input, Space } from 'antd';
import { Typography } from 'antd';

const { Text } = Typography;
import Sidebar, { type Session } from '../../components/Sidebar/Sidebar';
import MessageList, { type Message } from '../../components/MessageList/MessageList';
import ChatInput from '../../components/ChatInput/ChatInput';
import { chatApi } from '../../api/chat';
import { setToken } from '../../api/client';
import styles from './Chat.module.css';

// 生成唯一ID（会话 ID 用，身份 ID 由服务端签发）
const generateId = () => Math.random().toString(36).substring(2, 9);

const Chat: React.FC = () => {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [prefsOpen, setPrefsOpen] = useState(false);
  const [prefsActive, setPrefsActive] = useState<any[]>([]);
  const [prefsTimeline, setPrefsTimeline] = useState<any[]>([]);
  const [newPref, setNewPref] = useState('');
  // 加载会话列表
  const loadSessions = useCallback(async () => {
    try {
      const data = await chatApi.getSessions();
      const formattedSessions: Session[] = data.map((s) => ({
        id: s.session_id,
        title: `会话 ${s.session_id.slice(-6)}`,
        timestamp: s.last_active || s.created_at,
      }));
      setSessions(formattedSessions);
      return formattedSessions;
    } catch (error) {
      console.error('加载会话失败:', error);
      return [];
    }
  }, []);

  // 初始加载
  useEffect(() => {
    loadSessions().then((sessions) => {
      // 尝试从localStorage恢复当前会话
      const savedSessionId = localStorage.getItem('kbagent_current_session_id');
      if (savedSessionId) {
        // 检查该会话是否仍然存在
        const sessionExists = sessions.some((s) => s.id === savedSessionId);
        if (sessionExists) {
          handleSelectSession(savedSessionId);
        } else {
          // 会话已被删除，清除localStorage
          localStorage.removeItem('kbagent_current_session_id');
        }
      }
    });
  }, [loadSessions]);

  // 创建新会话
  const handleNewSession = () => {
    const newSessionId = `sess_${generateId()}`;
    const newSession: Session = {
      id: newSessionId,
      title: '新对话',
      timestamp: new Date().toISOString(),
    };
    setSessions((prev) => [newSession, ...prev]);
    setCurrentSessionId(newSessionId);
    localStorage.setItem('kbagent_current_session_id', newSessionId);
    setMessages([]);
  };

  // 选择会话
  const handleSelectSession = async (id: string) => {
    setCurrentSessionId(id);
    localStorage.setItem('kbagent_current_session_id', id);
    setMessages([]);

    try {
      const data = await chatApi.getSessionHistory(id);
      const formattedMessages: Message[] = [];

      data.history.forEach((turn, index) => {
        const timestamp = new Date().toISOString();
        formattedMessages.push({
          id: `msg_${index}_user`,
          role: 'user',
          content: turn.user,
          timestamp,
        });
        formattedMessages.push({
          id: `msg_${index}_assistant`,
          role: 'assistant',
          content: turn.assistant,
          timestamp,
        });
      });

      setMessages(formattedMessages);
    } catch (error) {
      console.error('加载会话历史失败:', error);
    }
  };

  // 删除会话
  const handleDeleteSession = async (id: string) => {
    try {
      await chatApi.deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));

      if (currentSessionId === id) {
        setCurrentSessionId(undefined);
        setMessages([]);
        localStorage.removeItem('kbagent_current_session_id');
      }

      antMessage.success('会话已删除');
    } catch (error) {
      antMessage.error('删除会话失败');
    }
  };

  // 发送消息（可选附图）
  const handleSendMessage = async (content: string, imageBase64?: string) => {
    // 没有当前会话时先创建一个
    let sessionId = currentSessionId;
    if (!sessionId) {
      const newSessionId = `sess_${generateId()}`;
      const title = (imageBase64 ? '[图片] ' : '') + content;
      const newSession: Session = {
        id: newSessionId,
        title: title.slice(0, 20) + (title.length > 20 ? '...' : ''),
        timestamp: new Date().toISOString(),
      };
      setSessions((prev) => [newSession, ...prev]);
      setCurrentSessionId(newSessionId);
      localStorage.setItem('kbagent_current_session_id', newSessionId);
      sessionId = newSessionId;
    }

    // 添加用户消息 + 空的助手占位（流式填充）
    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };
    const assistantId = generateId();
    setMessages((prev) => [
      ...prev,
      userMessage,
      { id: assistantId, role: 'assistant', content: '', streaming: true },
    ]);

    setLoading(true);
    let streamed = '';
    const finish = (finalReply?: string, contextCount?: number, toolUsed?: string) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: finalReply ?? streamed,
                streaming: false,
                contextCount,
                toolUsed,
              }
            : m,
        ),
      );
      setLoading(false);
    };

    // 优先流式；流式失败时 fallback 到非流式接口
    chatApi.sendMessageStream(
      { message: content, session_id: sessionId, image_base64: imageBase64 },
      {
        onToken: (t) => {
          streamed += t;
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: streamed } : m)),
          );
        },
        onDone: (payload) => {
          if (payload.token) {
            setToken(payload.token);
          }
          finish(payload.reply, payload.context_count, payload.tool_used);
        },
        onError: async () => {
          try {
            const resp = await chatApi.sendMessage({
              message: content,
              session_id: sessionId,
              image_base64: imageBase64,
            });
            if (resp.token) {
              setToken(resp.token);
            }
            finish(resp.reply, resp.context_count, resp.tool_used);
          } catch (err) {
            setMessages((prev) => prev.filter((m) => m.id !== assistantId));
            antMessage.error('发送消息失败，请重试');
            console.error('发送消息失败:', err);
            setLoading(false);
          }
        },
      },
    );
  };

  // 用户偏好管理（文件存储 + 时间线）
  const loadPrefs = useCallback(async () => {
    try {
      const data = await chatApi.getPreferences();
      setPrefsTimeline(data.timeline);
      setPrefsActive(data.active);
    } catch (e) {
      antMessage.error('加载偏好失败');
    }
  }, []);

  const handleAddPref = async () => {
    if (!newPref.trim()) return;
    try {
      await chatApi.addPreference(newPref.trim());
      setNewPref('');
      await loadPrefs();
      antMessage.success('已添加');
    } catch (e) {
      antMessage.error('添加失败');
    }
  };

  const handleUpdatePref = async (id: string, oldText: string) => {
    const next = window.prompt('修改偏好', oldText);
    if (!next || next.trim() === oldText) return;
    try {
      await chatApi.updatePreference(id, next.trim());
      await loadPrefs();
      antMessage.success('已更新');
    } catch (e) {
      antMessage.error('更新失败');
    }
  };

  const handleDeletePref = async (id: string) => {
    try {
      await chatApi.deletePreference(id);
      await loadPrefs();
      antMessage.success('已删除');
    } catch (e) {
      antMessage.error('删除失败');
    }
  };

  // 时间线里剔除当前生效项，剩下的就是"已被修改/删除"的历史（可回溯）
  const activePrefIds = new Set(prefsActive.map((item: any) => item.id));
  const prefsHistory = prefsTimeline.filter((item: any) => !activePrefIds.has(item.id));

  return (
    <div className={styles.chatPage}>
      <Sidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        onNewSession={handleNewSession}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
      />
      <div className={styles.chatContainer}>
        <div className={styles.chatHeader}>
          <h2>KBAgent 智能知识库问答</h2>
          <Button size="small" onClick={() => { loadPrefs(); setPrefsOpen(true); }}>
            我的偏好
          </Button>
        </div>
        <MessageList messages={messages} loading={loading} />
        <ChatInput onSend={handleSendMessage} loading={loading} />
      </div>
      <Modal
        title="我的偏好（时间线，矛盾会自动澄清）"
        open={prefsOpen}
        onCancel={() => setPrefsOpen(false)}
        footer={null}
        width={560}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <Space.Compact style={{ width: '100%' }}>
            <Input
              value={newPref}
              onChange={(e) => setNewPref(e.target.value)}
              placeholder="新增一条偏好，如：示例用 Python"
              onPressEnter={handleAddPref}
            />
            <Button type="primary" onClick={handleAddPref}>添加</Button>
          </Space.Compact>
          <List
            size="small"
            bordered
            dataSource={prefsActive}
            locale={{ emptyText: '暂无偏好' }}
            renderItem={(p: any) => (
              <List.Item
                actions={[
                  <a key="edit" onClick={() => handleUpdatePref(p.id, p.text)}>修改</a>,
                  <a key="del" onClick={() => handleDeletePref(p.id)} style={{ color: 'red' }}>删除</a>,
                ]}
              >
                <Text>（{String(p.updated_at).slice(0, 10)}）{p.text}</Text>
              </List.Item>
            )}
          />
          {prefsHistory.length > 0 && (
            <>
              <div style={{ marginTop: 8, color: '#999', fontSize: 13 }}>
                历史（已被修改/删除的旧偏好）
              </div>
              <List
                size="small"
                bordered
                dataSource={prefsHistory}
                renderItem={(item: any) => (
                  <List.Item>
                    <Text type="secondary">
                      （{String(item.updated_at).slice(0, 10)}）{item.text}
                    </Text>
                  </List.Item>
                )}
              />
            </>
          )}
        </Space>
      </Modal>
    </div>
  );
};

export default Chat;
