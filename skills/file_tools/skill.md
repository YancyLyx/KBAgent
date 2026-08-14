# File Tools Skill

## 描述
对项目文件进行读写操作，包括查看文件内容、创建新文件、编辑已有文件、列出目录。

## 工具

### read_file(path: str)
读取指定文件的内容。path 为相对于项目根目录的路径。

### write_file(path: str, content: str)
将内容写入指定文件（覆盖已有内容）。path 为相对于项目根目录的路径。

### create_file(path: str, content: str)
创建一个新文件并写入内容。如果文件已存在则返回错误。path 为相对于项目根目录的路径。

### list_files(path: str = ".")
列出指定目录下的文件和子目录。path 为相对于项目根目录的路径。

## 使用建议
- 读取文件用于获取上下文信息
- 创建和编辑文件前先确认项目结构
- 所有路径相对于项目根目录
