"""测试拆分历史功能"""

import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from yupoo_scraper.ml.splitter import check_split_history, extract_and_split
from yupoo_scraper.ml.split_history import get_split_history


def test_split_history():
    """测试拆分历史记录功能"""
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建测试文件夹
        test_folder = Path(tmpdir) / "test_folder"
        test_folder.mkdir()
        
        # 创建测试图片文件
        for i in range(3):
            img_path = test_folder / f"test{i}.jpg"
            img_path.write_text(f"test{i}")
        
        # 检查初始状态
        assert not check_split_history(test_folder), "初始状态应该未拆分"
        
        # 执行拆分
        result = extract_and_split(test_folder, threshold=0.5)
        
        # 检查拆分结果
        assert len(result.groups) > 0, "拆分应该生成分组"
        
        # 检查拆分历史
        assert check_split_history(test_folder), "拆分后应该记录历史"
        
        # 检查数据库记录
        history_db = get_split_history()
        record = history_db.get_split_history(str(test_folder))
        assert record is not None, "数据库应该有记录"
        assert record.group_count == len(result.groups), "分组数量应该一致"
        assert record.image_count == len(result.image_paths), "图片数量应该一致"
        assert record.has_features, "应该包含特征"
        
        # 检查分组信息
        groups = history_db.get_split_groups(record.id)
        assert len(groups) == len(result.groups), "分组信息数量应该一致"
        
        # 检查图片信息和特征
        for i, group in enumerate(result.groups):
            db_group_id, db_group_name = groups[i]
            assert db_group_name == group.name, "分组名称应该一致"
            
            db_image_paths = history_db.get_split_images(record.id, db_group_id)
            assert len(db_image_paths) == len(group.image_paths), "图片数量应该一致"
            
            # 检查特征
            for j, img_path in enumerate(group.image_paths):
                embedding = history_db.get_image_embedding(record.id, str(img_path))
                assert embedding is not None, "应该有特征嵌入"
                assert len(embedding) > 0, "特征应该不为空"
        
        print("测试通过！拆分历史功能正常工作。")


if __name__ == "__main__":
    test_split_history()
