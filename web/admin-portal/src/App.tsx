import React, { useState, useEffect } from 'react';
import { Layout, Menu, theme } from 'antd';
import {
  DashboardOutlined,
  FileTextOutlined,
  SettingOutlined,
  BarChartOutlined,
  UserOutlined,
} from '@ant-design/icons';
import Dashboard from './pages/Dashboard';
import Knowledge from './pages/Knowledge';
import Settings from './pages/Settings';
import Memory from './pages/Memory';
import Login from './pages/Login';
import { getAdminToken } from './api/client';

const { Header, Sider, Content } = Layout;

const App: React.FC = () => {
  const [authed, setAuthed] = useState<boolean>(() => Boolean(getAdminToken()));
  const [collapsed, setCollapsed] = useState(false);
  const onCollapse = () => setCollapsed(!collapsed);
  const [currentPage, setCurrentPage] = useState('dashboard');

  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken();

  // token 失效（401）时回到登录页
  useEffect(() => {
    const onUnauthorized = () => setAuthed(false);
    window.addEventListener('admin:unauthorized', onUnauthorized);
    return () => window.removeEventListener('admin:unauthorized', onUnauthorized);
  }, []);

  if (!authed) {
    return <Login onLoginSuccess={() => setAuthed(true)} />;
  }

  const menuItems = [
    { key: 'dashboard', icon: <DashboardOutlined />, label: '仪表盘' },
    { key: 'knowledge', icon: <FileTextOutlined />, label: '知识库' },
    { key: 'settings', icon: <SettingOutlined />, label: '策略配置' },
    { key: 'memory', icon: <UserOutlined />, label: '用户记忆' },
    { key: 'monitor', icon: <BarChartOutlined />, label: '系统监控' },
  ];

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />;
      case 'knowledge':
        return <Knowledge />;
      case 'settings':
        return <Settings />;
      case 'memory':
        return <Memory />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider collapsible collapsed={collapsed} onCollapse={onCollapse}>
        <div style={{ height: 64, margin: 16, background: 'rgba(255,255,255,0.1)', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontSize: collapsed ? 14 : 18, fontWeight: 'bold' }}>
          {collapsed ? 'KB' : 'KBAgent'}
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[currentPage]}
          items={menuItems}
          onClick={({ key }) => setCurrentPage(key)}
        />
      </Sider>

      <Layout>
        <Header style={{ padding: 0, background: colorBgContainer }} >
          <div style={{ padding: '0 24px', fontSize: 18, fontWeight: 'bold' }}>
            KBAgent 管理后台
          </div>
        </Header>

        <Content
          style={{
            margin: '24px 16px',
            padding: 24,
            minHeight: 280,
            background: colorBgContainer,
            borderRadius: borderRadiusLG,
          }}
        >
          {renderPage()}
        </Content>
      </Layout>
    </Layout>
  );
};

export default App;
