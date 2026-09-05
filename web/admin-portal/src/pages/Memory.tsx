import React, { useState } from 'react';
import {
  Card,
  Input,
  Button,
  Table,
  Tag,
  Typography,
  Space,
  Empty,
  message,
  Popconfirm,
} from 'antd';
import { SearchOutlined, DeleteOutlined, ClearOutlined } from '@ant-design/icons';
import { memoryApi, UserMemory } from '../api/memory';

const { Title, Paragraph, Text } = Typography;

const Memory: React.FC = () => {
  const [userId, setUserId] = useState('');
  const [data, setData] = useState<UserMemory | null>(null);
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState('');

  const handleSearch = async () => {
    const uid = userId.trim();
    if (!uid) {
      message.warning('请输入 user_id');
      return;
    }
    setLoading(true);
    try {
      const resp = await memoryApi.getMemory(uid);
      setData(resp);
      setSearching(uid);
    } catch (err: any) {
      message.error(err.response?.data?.detail || '查询失败');
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteFact = async (fact: string) => {
    try {
      await memoryApi.deleteFact(searching, fact);
      message.success('已删除该画像条目');
      handleSearch();
    } catch (err: any) {
      message.error(err.response?.data?.detail || '删除失败');
    }
  };

  const handleDeleteManualPref = async (prefId: string) => {
    try {
      await memoryApi.deleteManualPref(searching, prefId);
      message.success('已删除该手动偏好');
      handleSearch();
    } catch (err: any) {
      message.error(err.response?.data?.detail || '删除失败');
    }
  };

  const handleClear = async () => {
    try {
      await memoryApi.clearMemory(searching);
      message.success('用户记忆已清空');
      setData(null);
    } catch (err: any) {
      message.error(err.response?.data?.detail || '清空失败');
    }
  };

  const facts = data?.profile?.preferences?.extracted_facts ?? [];
  const manualPrefs = data?.prefs_active ?? [];
  const timeline = data?.prefs_timeline ?? [];
  // 时间线里剔除当前生效项 = 已被修改/删除的历史（可审计）
  const activeIds = new Set(manualPrefs.map((p) => p.id));
  const prefHistory = timeline.filter((p) => !activeIds.has(p.id));

  const fmtTime = (t?: string) =>
    t ? t.slice(0, 19).replace('T', ' ') : '-';

  return (
    <div>
      <Title level={4}>用户记忆管理</Title>
      <Paragraph type="secondary">
        查看/纠错用户自动画像（对话提取）、手动偏好（前端「我的偏好」添加）、演进式摘要与审计日志。
      </Paragraph>

      <Space.Compact style={{ marginBottom: 16, width: 480 }}>
        <Input
          placeholder="输入 user_id（如 user_8969a39e1546）"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          onPressEnter={handleSearch}
          allowClear
        />
        <Button type="primary" icon={<SearchOutlined />} onClick={handleSearch} loading={loading}>
          查询
        </Button>
      </Space.Compact>

      {!data ? (
        <Empty description="输入 user_id 查询用户记忆" />
      ) : (
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <Card
            title="用户自动画像（对话中提取）"
            extra={
              <Popconfirm
                title="确认清空该用户全部记忆？"
                description="将删除自动画像、演进式摘要与话题计数（手动偏好与审计日志保留）"
                onConfirm={handleClear}
                okText="确认清空"
                cancelText="取消"
              >
                <Button danger icon={<ClearOutlined />} size="small">
                  清空记忆
                </Button>
              </Popconfirm>
            }
          >
            {facts.length === 0 ? (
              <Empty description="暂无自动提取画像（手动偏好在下方卡片查看）" />
            ) : (
              <Table
                rowKey="text"
                size="small"
                pagination={false}
                dataSource={facts}
                columns={[
                  {
                    title: '偏好条目',
                    dataIndex: 'text',
                    render: (t: string) => <Text strong>{t}</Text>,
                  },
                  {
                    title: '更新时间',
                    dataIndex: 'updated_at',
                    width: 180,
                    render: fmtTime,
                  },
                  {
                    title: '来源',
                    dataIndex: 'source',
                    width: 160,
                    render: (s?: string) =>
                      s ? <Tag color="blue">{s}</Tag> : <Tag>auto</Tag>,
                  },
                  {
                    title: '操作',
                    width: 100,
                    render: (_: unknown, record: { text: string }) => (
                      <Popconfirm
                        title="删除该画像条目？"
                        onConfirm={() => handleDeleteFact(record.text)}
                      >
                        <Button danger size="small" icon={<DeleteOutlined />}>
                          删除
                        </Button>
                      </Popconfirm>
                    ),
                  },
                ]}
              />
            )}
          </Card>

          <Card
            title="手动偏好（前端「我的偏好」添加）"
            extra={
              <Tag color={manualPrefs.length > 0 ? 'green' : 'default'}>
                当前生效 {manualPrefs.length} 条
              </Tag>
            }
          >
            {manualPrefs.length === 0 ? (
              <Empty description="暂无手动偏好" />
            ) : (
              <Table
                rowKey="id"
                size="small"
                pagination={false}
                dataSource={manualPrefs}
                columns={[
                  {
                    title: '偏好内容',
                    dataIndex: 'text',
                    render: (t: string) => <Text strong>{t}</Text>,
                  },
                  {
                    title: '添加时间',
                    dataIndex: 'updated_at',
                    width: 180,
                    render: fmtTime,
                  },
                  {
                    title: '来源',
                    dataIndex: 'source',
                    width: 140,
                    render: (s?: string) =>
                      s ? <Tag color="purple">{s}</Tag> : <Tag>manual</Tag>,
                  },
                  {
                    title: '操作',
                    width: 100,
                    render: (_: unknown, record: { id: string }) => (
                      <Popconfirm
                        title="软删除这条手动偏好？"
                        description="历史时间线会保留，只是不再注入"
                        onConfirm={() => handleDeleteManualPref(record.id)}
                      >
                        <Button danger size="small" icon={<DeleteOutlined />}>
                          删除
                        </Button>
                      </Popconfirm>
                    ),
                  },
                ]}
              />
            )}
            {prefHistory.length > 0 && (
              <Paragraph style={{ marginTop: 12, color: '#999', marginBottom: 0 }}>
                历史（已被修改/删除，不再注入）：
                <br />
                {prefHistory.map((p) => (
                  <span key={p.id}>
                    - {fmtTime(p.updated_at)}：{p.text || '（已删除）'}{' '}
                    {p.status === 'deleted' ? <Tag>deleted</Tag> : <Tag>superseded</Tag>}
                    <br />
                  </span>
                ))}
              </Paragraph>
            )}
          </Card>

          <Card title="演进式摘要">
            <Paragraph style={{ whiteSpace: 'pre-wrap' }}>
              {data.running_summary || '（暂无摘要）'}
            </Paragraph>
          </Card>

          <Card title="审计日志（变更历史，人可读）">
            <pre
              style={{
                maxHeight: 320,
                overflow: 'auto',
                background: '#fafafa',
                padding: 12,
                fontSize: 12,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-all',
              }}
            >
              {data.audit_md || '（暂无变更记录）'}
            </pre>
          </Card>
        </Space>
      )}
    </div>
  );
};

export default Memory;
