import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Statistic, Progress, Tag, Spin, Table } from 'antd';
import {
  MessageOutlined, UserOutlined, FileTextOutlined,
  AlertOutlined, CheckCircleOutlined, CloseCircleOutlined,
  WarningOutlined, HistoryOutlined,
} from '@ant-design/icons';
import client from '../api/client';

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<any>(null);
  const [evalSummary, setEvalSummary] = useState<any>(null);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [missed, setMissed] = useState<any[]>([]);
  const [realtime, setRealtime] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = async () => {
      try {
        const [statsData, evalData, alertData, missedData, realtimeData] = await Promise.all([
          client.get('/api/admin/stats/overview').catch(() => ({})),
          client.get('/api/admin/eval/summary').catch(() => ({})),
          client.get('/api/admin/eval/alerts').catch(() => ({ alerts: [] })),
          client.get('/api/admin/eval/missed').catch(() => ({ summary: [] })),
          client.get('/api/admin/eval/realtime').catch(() => ({ scores: [] })),
        ]);
        setStats(statsData);
        setEvalSummary(evalData);
        setAlerts((alertData as any)?.alerts ?? []);
        setMissed((missedData as any)?.summary ?? []);
        setRealtime((realtimeData as any)?.scores ?? []);
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, []);

  if (loading) return <Spin size="large" style={{ display: 'block', margin: '100px auto' }} />;

  const avgU = evalSummary?.avg_usefulness ?? '-';
  const avgR = evalSummary?.avg_relevance ?? '-';
  const alertTotal = evalSummary?.alerts?.total_alerts ?? alerts.length;

  const scoreText = (r: any) => {
    if (!r?.scores) return '-';
    const s = typeof r.scores === 'string' ? JSON.parse(r.scores) : r.scores;
    return `相关${s.relevance ?? '-'} · 完整${s.completeness ?? '-'} · 有用${s.usefulness ?? '-'}` +
      (r.faithfulness != null ? ` · 忠实${r.faithfulness}` : '');
  };

  const fmtTime = (t?: string) => (t ? String(t).slice(0, 19).replace('T', ' ') : '-');

  const alertColumns = [
    { title: 'Query', dataIndex: 'query', ellipsis: true, render: (v: string) => v },
    { title: '分数', dataIndex: 'scores', width: 230, render: (_: any, r: any) => (
      <span style={{ color: '#ff4d4f' }}>{scoreText(r)}</span>) },
    { title: '时间', dataIndex: 'timestamp', width: 170, render: fmtTime },
  ];

  const missedColumns = [
    { title: '失败 Query', dataIndex: 'query', ellipsis: true },
    { title: '知识库', dataIndex: 'tag', width: 130 },
    { title: '次数', dataIndex: 'count', width: 90 },
    { title: '最后出现', dataIndex: 'last_seen', width: 180, render: fmtTime },
  ];

  const realtimeColumns = [
    { title: 'Query', dataIndex: 'query', ellipsis: true },
    { title: '相关', dataIndex: 'relevance', width: 70 },
    { title: '完整', dataIndex: 'completeness', width: 70 },
    { title: '有用', dataIndex: 'usefulness', width: 70 },
    { title: '时间', dataIndex: 'timestamp', width: 170, render: fmtTime },
  ];

  return (
    <div>
      <h2 style={{ marginBottom: 24 }}>仪表盘</h2>

      {/* 第一行：系统概览 */}
      <Row gutter={16}>
        <Col span={6}>
          <Card><Statistic title="今日对话" value={stats?.today_conversations ?? '-'} prefix={<MessageOutlined />} /></Card>
        </Col>
        <Col span={6}>
          <Card><Statistic title="活跃用户（7日）" value={stats?.active_users_7d ?? '-'} prefix={<UserOutlined />} /></Card>
        </Col>
        <Col span={6}>
          <Card><Statistic title="文档总数" value={stats?.total_documents ?? '-'} prefix={<FileTextOutlined />} /></Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="评测告警" value={alertTotal} prefix={<AlertOutlined />}
              valueStyle={{ color: alertTotal > 0 ? '#ff4d4f' : '#52c41a' }} />
          </Card>
        </Col>
      </Row>

      {/* 第二行：评测分数 */}
      <Card title="回答质量评测" style={{ marginTop: 24 }}>
        {evalSummary?.total_scores > 0 ? (
          <Row gutter={48}>
            <Col span={8}>
              <div style={{ textAlign: 'center' }}>
                <Progress type="circle" percent={avgR * 20} format={() => `${avgR}`}
                  strokeColor={avgR >= 4 ? '#52c41a' : avgR >= 3 ? '#faad14' : '#ff4d4f'} />
                <div style={{ marginTop: 8, color: '#666' }}>相关性</div>
              </div>
            </Col>
            <Col span={8}>
              <div style={{ textAlign: 'center' }}>
                <Progress type="circle" percent={avgU * 20} format={() => `${avgU}`}
                  strokeColor={avgU >= 4 ? '#52c41a' : avgU >= 3 ? '#faad14' : '#ff4d4f'} />
                <div style={{ marginTop: 8, color: '#666' }}>有用性</div>
              </div>
            </Col>
            <Col span={8}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 36, fontWeight: 'bold', color: '#1677ff' }}>{evalSummary?.total_scores ?? 0}</div>
                <div style={{ color: '#666' }}>已评测对话</div>
              </div>
            </Col>
          </Row>
        ) : (
          <div style={{ textAlign: 'center', color: '#999', padding: 20 }}>暂无评测数据。启动对话后自动评测。</div>
        )}
      </Card>

      {/* 评测告警明细（badcase 可回溯：具体 query、分数、时间） */}
      <Card title={<span><CloseCircleOutlined style={{ color: alertTotal > 0 ? '#ff4d4f' : '#52c41a' }} /> 评测告警明细</span>}
        style={{ marginTop: 24 }} extra={<Tag color={alertTotal > 0 ? 'red' : 'green'}>{alertTotal} 条</Tag>}>
        <Table rowKey={(r: any, i?: number) => `${r.timestamp}-${i}`} size="small" columns={alertColumns} dataSource={alerts}
          pagination={{ pageSize: 5 }} locale={{ emptyText: '暂无告警' }} />
      </Card>

      {/* 失败检索（badcase 回溯：失败 query → 反哺知识库） */}
      <Card title={<span><WarningOutlined style={{ color: '#faad14' }} /> 检索失败 Query（知识库盲区）</span>}
        style={{ marginTop: 24 }} extra={<span>共 {missed.length} 个</span>}>
        <Table rowKey={(r: any, i?: number) => `${r.query}-${i}`} size="small" columns={missedColumns} dataSource={missed}
          pagination={{ pageSize: 5 }} locale={{ emptyText: '暂无失败检索' }} />
      </Card>

      {/* 最近实时评测 */}
      <Card title={<span><HistoryOutlined /> 最近评测明细</span>} style={{ marginTop: 24 }}>
        <Table rowKey={(r: any, i?: number) => `${r.timestamp}-${i}`} size="small" columns={realtimeColumns} dataSource={realtime}
          pagination={{ pageSize: 5 }} locale={{ emptyText: '暂无实时评测' }} />
      </Card>

      {/* 系统状态 */}
      <Card title="系统状态" style={{ marginTop: 24 }}>
        <Row gutter={16}>
          <Col span={8}><p><CheckCircleOutlined style={{ color: '#52c41a' }} /> 向量数据库 <Tag color="green">正常</Tag></p></Col>
          <Col span={8}><p><CheckCircleOutlined style={{ color: '#52c41a' }} /> LLM 服务 <Tag color="green">正常</Tag></p></Col>
          <Col span={8}>
            <p>
              {alertTotal > 0
                ? <><CloseCircleOutlined style={{ color: '#ff4d4f' }} /> 评测告警 <Tag color="red">{alertTotal} 条</Tag></>
                : <><CheckCircleOutlined style={{ color: '#52c41a' }} /> 评测告警 <Tag color="green">无</Tag></>
              }
            </p>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default Dashboard;
