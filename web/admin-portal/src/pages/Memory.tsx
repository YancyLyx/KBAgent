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
      message.success('已删除该偏好条目');
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

  return (
    <div>
      <Title level={4}>用户记忆管理</Title>
      <Paragraph type="secondary">
        查看/纠错用户画像、演进式摘要与审计日志（记忆被错误信息污染时的人工修正入口）。
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
            title="用户画像"
            extra={
              <Popconfirm
                title="确认清空该用户全部记忆？"
                description="将删除画像、演进式摘要与话题计数（审计日志保留）"
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
              <Empty description="暂无偏好条目" />
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
                    render: (t?: string) => (t ? t.slice(0, 19).replace('T', ' ') : '-'),
                  },
                  {
                    title: '来源',
                    dataIndex: 'source',
                    width: 160,
                    render: (s?: string) =>
                      s ? <Tag color="blue">{s}</Tag> : <Tag>manual</Tag>,
                  },
                  {
                    title: '操作',
                    width: 100,
                    render: (_: unknown, record: { text: string }) => (
                      <Popconfirm
                        title="删除该偏好条目？"
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
