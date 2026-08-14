import React, { useState, useEffect } from 'react';
import { Table, Button, Upload, Input, message, Space, Modal, Spin, Popconfirm } from 'antd';
import { UploadOutlined, DeleteOutlined, EyeOutlined, ReloadOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';
import { documentApi, Document } from '../api/documents';
import client from '../api/client';

const Knowledge: React.FC = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewContent, setPreviewContent] = useState('');
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewTitle, setPreviewTitle] = useState('');
  const [uploadTag, setUploadTag] = useState('');

  // 加载文档列表
  const loadDocuments = async () => {
    setLoading(true);
    try {
      const data = await documentApi.getList(1, 50);
      setDocuments(data.items);
    } catch (error) {
      message.error('加载文档列表失败');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // 初始加载
  useEffect(() => {
    loadDocuments();
  }, []);

  // 加载现有标签
  useEffect(() => {
    documentApi.getTags().then((tags) => {
      // 标签已由上传表单选择使用，这里仅预加载（保留给后续筛选功能）
      void tags;
    });
  }, []);

  // 删除文档
  const handleDelete = async (id: string) => {
    try {
      await documentApi.delete(id);
      message.success('文档已删除');
      // 刷新列表
      loadDocuments();
    } catch (error) {
      message.error('删除文档失败');
      console.error(error);
    }
  };

  // 预览文档
  const handlePreview = async (record: Document) => {
    setPreviewVisible(true);
    setPreviewLoading(true);
    setPreviewTitle(record.original_name);
    try {
      const data = await documentApi.getContent(record.id);
      setPreviewContent(data.content);
    } catch (error) {
      message.error('加载文档内容失败');
      setPreviewContent('加载失败');
    } finally {
      setPreviewLoading(false);
    }
  };

  // 重新索引
  const handleReindex = async (id: string) => {
    try {
      await documentApi.reindex(id);
      message.success('重新索引任务已启动');
    } catch (error) {
      message.error('重新索引失败');
    }
  };

  const uploadProps: UploadProps = {
    name: 'file',
    customRequest: async ({ file, onSuccess, onError }) => {
      const formData = new FormData();
      formData.append('file', file as Blob);
      formData.append('tag', uploadTag);

      try {
        const resp = await fetch(`${client.defaults.baseURL}/api/admin/documents/upload`, {
          method: 'POST',
          body: formData,
        });
        if (!resp.ok) {
          const errData = await resp.json().catch(() => ({}));
          throw new Error(errData.detail || `上传失败 (${resp.status})`);
        }
        const data = await resp.json();
        onSuccess?.(data);
        message.success(`${(file as File).name} 上传成功`);
        loadDocuments();
      } catch (err: any) {
        onError?.(err);
        message.error(`${(file as File).name} 上传失败: ${err.message}`);
      }
    },
  };

  // 格式化文件大小
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // 格式化时间
  const formatTime = (time: string) => {
    return new Date(time).toLocaleString('zh-CN');
  };

  const columns = [
    { title: '文件名', dataIndex: 'original_name', key: 'original_name' },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => formatFileSize(size)
    },
    {
      title: '标签',
      dataIndex: 'tag',
      key: 'tag',
      render: (tag: string) => tag ? <span style={{ color: '#1677ff' }}>{tag}</span> : '-',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (s: string) => {
        const statusMap: Record<string, { text: string; color: string }> = {
          pending: { text: '待处理', color: 'orange' },
          processing: { text: '处理中', color: 'blue' },
          indexed: { text: '已索引', color: 'green' },
          error: { text: '错误', color: 'red' },
        };
        const status = statusMap[s] || { text: s, color: 'default' };
        return <span style={{ color: status.color }}>{status.text}</span>;
      }
    },
    {
      title: '上传时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => formatTime(time)
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: Document) => (
        <Space>
          <Button
            icon={<EyeOutlined />}
            size="small"
            onClick={() => handlePreview(record)}
          >预览</Button>
          <Button
            icon={<ReloadOutlined />}
            size="small"
            onClick={() => handleReindex(record.id)}
          >重索引</Button>
          <Popconfirm
            title="确定删除这个文档吗？"
            description="删除后将无法恢复"
            onConfirm={() => handleDelete(record.id)}
            okText="确定"
            cancelText="取消"
          >
            <Button
              icon={<DeleteOutlined />}
              danger
              size="small"
            >删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <h2 style={{ marginBottom: 24 }}>知识库管理</h2>

      <div style={{ marginBottom: 16, display: 'flex', gap: 12 }}>
        <Input
          placeholder="标签（可选）"
          value={uploadTag}
          onChange={(e) => setUploadTag(e.target.value)}
          style={{ width: 260 }}
          allowClear
        />
        <Upload {...uploadProps}>
          <Button icon={<UploadOutlined />}>上传文档</Button>
        </Upload>
        <Button onClick={loadDocuments} loading={loading}>刷新列表</Button>
      </div>

      <Table
        columns={columns}
        dataSource={documents}
        rowKey="id"
        loading={loading}
        pagination={{ pageSize: 10 }}
      />

      {/* 预览弹窗 */}
      <Modal
        title={previewTitle}
        open={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        footer={null}
        width={800}
      >
        {previewLoading ? (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Spin />
          </div>
        ) : (
          <pre style={{
            maxHeight: 500,
            overflow: 'auto',
            backgroundColor: '#f6f8fa',
            padding: 16,
            borderRadius: 8
          }}>
            {previewContent}
          </pre>
        )}
      </Modal>
    </div>
  );
};

export default Knowledge;
