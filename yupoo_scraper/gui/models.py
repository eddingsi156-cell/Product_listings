"""虚拟表格模型 — 支持万级行数无卡顿

替代 QTableWidget，仅渲染可见行。
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QStyle,
    QStyleOptionButton,
    QStyledItemDelegate,
    QApplication,
    QWidget,
)


# ── 自定义角色 ─────────────────────────────────────────────────

ROLE_ACTION_TYPE = Qt.ItemDataRole.UserRole + 1    # str: "button", "label"
ROLE_ACTION_TEXT = Qt.ItemDataRole.UserRole + 2    # str: 按钮/标签文字
ROLE_ACTION_COLOR = Qt.ItemDataRole.UserRole + 3   # str: CSS 颜色（仅 label）
ROLE_ACTION_TOOLTIP = Qt.ItemDataRole.UserRole + 4 # str: tooltip
ROLE_RAW_DATA = Qt.ItemDataRole.UserRole + 10      # Any: 原始数据对象


# ── VirtualTableModel ─────────────────────────────────────────

class VirtualTableModel(QAbstractTableModel):
    """通用虚拟表格模型基类。

    子类需要实现:
    - _headers: list[str]          列标题
    - _column_data(row, col, role): 返回单元格数据
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[Any] = []

    @property
    def _headers(self) -> list[str]:
        raise NotImplementedError

    def rowCount(self, parent=QModelIndex()):
        return len(self._items)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            if 0 <= section < len(self._headers):
                return self._headers[section]
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        if row < 0 or row >= len(self._items):
            return None
        return self._column_data(row, col, role)

    def _column_data(self, row: int, col: int, role: int) -> Any:
        """子类实现：返回指定单元格的数据。"""
        raise NotImplementedError

    def flags(self, index: QModelIndex):
        flags = super().flags(index)
        # 移除可编辑标志
        flags &= ~Qt.ItemFlag.ItemIsEditable
        return flags

    # ── 数据操作 API ──────────────────────────────────────────

    def get_item(self, row: int) -> Any:
        """获取原始数据对象。"""
        if 0 <= row < len(self._items):
            return self._items[row]
        return None

    def append_item(self, item: Any) -> None:
        """追加一行。"""
        row = len(self._items)
        self.beginInsertRows(QModelIndex(), row, row)
        self._items.append(item)
        self.endInsertRows()

    def set_item(self, row: int, item: Any) -> None:
        """替换指定行的数据。"""
        if 0 <= row < len(self._items):
            self._items[row] = item
            left = self.index(row, 0)
            right = self.index(row, self.columnCount() - 1)
            self.dataChanged.emit(left, right)

    def remove_item(self, row: int) -> None:
        """删除指定行。"""
        if 0 <= row < len(self._items):
            self.beginRemoveRows(QModelIndex(), row, row)
            self._items.pop(row)
            self.endRemoveRows()

    def clear(self) -> None:
        """清空所有数据。"""
        if self._items:
            self.beginResetModel()
            self._items.clear()
            self.endResetModel()

    def reset_items(self, items: list[Any]) -> None:
        """批量替换所有数据。"""
        self.beginResetModel()
        self._items = list(items)
        self.endResetModel()

    @property
    def items(self) -> list[Any]:
        return self._items


# ── ButtonDelegate ─────────────────────────────────────────────

class ButtonDelegate(QStyledItemDelegate):
    """操作列委托：根据模型数据绘制按钮或标签，处理点击。

    模型需通过自定义角色返回:
    - ROLE_ACTION_TYPE: "button" 或 "label"
    - ROLE_ACTION_TEXT: 显示文字
    - ROLE_ACTION_COLOR: 标签颜色 (可选)
    - ROLE_ACTION_TOOLTIP: 提示文字 (可选)
    """

    def __init__(self, on_click: Callable[[int], None] | None = None, parent=None):
        super().__init__(parent)
        self._on_click = on_click

    def paint(self, painter, option, index):
        action_type = index.data(ROLE_ACTION_TYPE)
        text = index.data(ROLE_ACTION_TEXT) or ""

        if action_type == "button":
            # 绘制按钮外观
            btn_opt = QStyleOptionButton()
            btn_opt.rect = option.rect.adjusted(4, 2, -4, -2)
            btn_opt.text = text
            btn_opt.state = QStyle.StateFlag.State_Enabled
            if option.state & QStyle.StateFlag.State_MouseOver:
                btn_opt.state |= QStyle.StateFlag.State_MouseOver
            QApplication.style().drawControl(
                QStyle.ControlElement.CE_PushButton, btn_opt, painter
            )
        else:
            # 绘制标签
            painter.save()
            color_str = index.data(ROLE_ACTION_COLOR)
            if color_str:
                painter.setPen(QColor(color_str))
            painter.drawText(
                option.rect, Qt.AlignmentFlag.AlignCenter, text
            )
            painter.restore()

    def editorEvent(self, event, model, option, index):
        from PySide6.QtCore import QEvent
        if (event.type() == QEvent.Type.MouseButtonRelease
                and index.data(ROLE_ACTION_TYPE) == "button"
                and self._on_click is not None):
            # 如果使用了 proxy model，需要映射回源行号
            source_index = index
            if hasattr(model, 'mapToSource'):
                source_index = model.mapToSource(index)
            self._on_click(source_index.row())
            return True
        return False

    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        hint.setHeight(max(hint.height(), 28))
        return hint
