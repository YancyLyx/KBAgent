import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Statistic, Progress, Tag, Spin } from 'antd';
import {
  MessageOutlined, UserOutlined, FileTextOutlined,
  AlertOutlined, CheckCircleOutlined, CloseCircleOutlined,
} from '@ant-design/icons';
import client from '../api/client';

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<any>(null);
  const [evalSummary, setEvalSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = async () => {
      try {
        const [statsData, evalData] = await Promise.all([
          client.get('/api/admin/stats/overview').catch(() => ({})),
          client.get('/api/admin/eval/summary').catch(() => ({})),
        ]);
        setStats(statsData);
        setEvalSummary(evalData);
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, []);

  if (loading) return <Spin size="large" style={{ display: 'block', margin: '100px auto' }} />;

  const avgU = evalSummary?.avg_usefulness ?? '-';
  const avgR = evalSummary?.avg_relevance ?? '-';
  const alertTotal = evalSummary?.alerts?.total_alerts ?? 0;

  return (
    <div>
      <h2 style={{ marginBottom: 24 }}>仪表盘</h2>

      {/* 第一行：系统概览 */}
      <Row gutter={16}>
        <Col span={6}>
          <Card>
            <Statistic title="今日对话" value={stats?.today_conversations ?? '-'}
              prefix={<MessageOutlined />} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="活跃用户（7日）" value={stats?.active_users_7d ?? '-'}
              prefix={<UserOutlined />} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="文档总数" value={stats?.total_documents ?? '-'}
              prefix={<FileTextOutlined />} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="评测告警" value={alertTotal}
              prefix={<AlertOutlined />}
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
                <Progress type="circle" percent={avgR * 20}
                  format={() => `${avgR}`}
                  strokeColor={avgR >= 4 ? '#52c41a' : avgR >= 3 ? '#faad14' : '#ff4d4f'} />
                <div style={{ marginTop: 8, color: '#666' }}>相关性</div>
              </div>
            </Col>
            <Col span={8}>
              <div style={{ textAlign: 'center' }}>
                <Progress type="circle" percent={avgU * 20}
                  format={() => `${avgU}`}
                  strokeColor={avgU >= 4 ? '#52c41a' : avgU >= 3 ? '#faad14' : '#ff4d4f'} />
                <div style={{ marginTop: 8, color: '#666' }}>有用性</div>
              </div>
            </Col>
            <Col span={8}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 36, fontWeight: 'bold', color: '#1677ff' }}>
                  {evalSummary?.total_scores ?? 0}
                </div>
                <div style={{ color: '#666' }}>已评测对话</div>
              </div>
            </Col>
          </Row>
        ) : (
          <div style={{ textAlign: 'center', color: '#999', padding: 20 }}>
            暂无评测数据。启动对话后自动评测。
          </div>
        )}
      </Card>

      {/* 系统状态 */}
      <Card title="系统状态" style={{ marginTop: 24 }}>
        <Row gutter={16}>
          <Col span={8}>
            <p><CheckCircleOutlined style={{ color: '#52c41a' }} /> 向量数据库 <Tag color="green">正常</Tag></p>
          </Col>
          <Col span={8}>
            <p><CheckCircleOutlined style={{ color: '#52c41a' }} /> LLM 服务 <Tag color="green">正常</Tag></p>
          </Col>
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
