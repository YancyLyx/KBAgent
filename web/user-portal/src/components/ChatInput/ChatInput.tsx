import React, { useState, useRef, type KeyboardEvent, type ChangeEvent } from 'react';
import { Input, Button, Image } from 'antd';
import { SendOutlined, PictureOutlined, CloseCircleOutlined } from '@ant-design/icons';
import styles from './ChatInput.module.css';

const { TextArea } = Input;

interface ChatInputProps {
  onSend: (message: string, imageBase64?: string) => void;
  loading?: boolean;
  disabled?: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSend, loading, disabled }) => {
  const [message, setMessage] = useState('');
  const [imageBase64, setImageBase64] = useState<string | undefined>();
  const fileRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<any>(null);

  const handleFile = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = ''; // 允许重复选择同一文件
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setImageBase64(String(reader.result));
    reader.readAsDataURL(file);
  };

  const handleSend = () => {
    const trimmed = message.trim();
    if ((!trimmed && !imageBase64) || loading || disabled) return;
    // 只发图没文字时补一句默认指令（后端 message 必填）
    onSend(trimmed || '请查看这张图片并回答', imageBase64);
    setMessage('');
    setImageBase64(undefined);
    if (inputRef.current) {
      inputRef.current.resizableTextArea.textArea.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={styles.chatInput}>
      <div className={styles.inputWrapper}>
        {imageBase64 && (
          <div style={{ position: 'relative', display: 'inline-block', marginBottom: 4 }}>
            <Image
              src={imageBase64}
              alt="附图"
              width={72}
              height={72}
              style={{ objectFit: 'cover', borderRadius: 6, border: '1px solid #d9d9d9' }}
            />
            <Button
              size="small"
              type="text"
              icon={<CloseCircleOutlined />}
              onClick={() => setImageBase64(undefined)}
              style={{ position: 'absolute', top: -8, right: -8, color: '#ff4d4f' }}
            />
          </div>
        )}
        <input ref={fileRef} type="file" accept="image/png,image/jpeg,image/webp,image/bmp" style={{ display: 'none' }} onChange={handleFile} />
        <Button
          icon={<PictureOutlined />}
          onClick={() => fileRef.current?.click()}
          disabled={disabled || loading}
          style={{ marginRight: 4 }}
          title="上传图片（图表/截图，VLM 理解后回答）"
        />
        <TextArea
          ref={inputRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="请输入您的问题...（支持上传图片，Shift+Enter换行）"
          autoSize={{ minRows: 1, maxRows: 4 }}
          disabled={disabled || loading}
          className={styles.textArea}
        />
        <Button
          type="primary"
          icon={<SendOutlined />}
          onClick={handleSend}
          loading={loading}
          disabled={(!message.trim() && !imageBase64) || disabled}
          className={styles.sendButton}
        >
          发送
        </Button>
      </div>
    </div>
  );
};

export default ChatInput;
