import sys
import re
import inspect
import h5py
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTreeWidget,
    QTreeWidgetItem, QSplitter, QListWidget, QListWidgetItem,
    QPushButton, QDialog, QDialogButtonBox, QLabel, QLineEdit,
    QFormLayout, QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem,
    QHBoxLayout, QMenu
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QAction

from slothpy.core._registry import type_registry
from slothpy.core._slt_file import SltGroup


##############################################################################
# 1) ANSI-to-HTML color conversion
##############################################################################

RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"
PURPLE = "\033[35m"
RESET  = "\033[0m"

ANSI_COLORS = {
    "\033[31m": '<span style="color:red">',
    "\033[32m": '<span style="color:green">',
    "\033[33m": '<span style="color:gold">',
    "\033[34m": '<span style="color:blue">',
    "\033[35m": '<span style="color:purple">',
    "\033[0m":  '</span>',
}
ANSI_PATTERN = re.compile(
    r"(?:\033\[31m|\033\[32m|\033\[33m|\033\[34m|\033\[35m|\033\[0m)"
)

def ansi_to_html(ansi_text: str) -> str:
    """Convert ANSI color codes to <span> tags (HTML)."""
    def replace_code(match):
        code = match.group(0)
        return ANSI_COLORS.get(code, '')
    html_text = ANSI_PATTERN.sub(replace_code, ansi_text)
    return f"<html><body>{html_text}</body></html>"

def show_colored_info(title: str, ansi_text: str):
    mbox = QMessageBox()
    mbox.setIcon(QMessageBox.Icon.Information)
    mbox.setWindowTitle(title)
    mbox.setTextFormat(Qt.TextFormat.RichText)
    mbox.setText(ansi_to_html(ansi_text))
    mbox.exec()

def show_colored_warning(title: str, ansi_text: str):
    mbox = QMessageBox()
    mbox.setIcon(QMessageBox.Icon.Warning)
    mbox.setWindowTitle(title)
    mbox.setTextFormat(Qt.TextFormat.RichText)
    mbox.setText(ansi_to_html(ansi_text))
    mbox.exec()

def show_colored_error(title: str, ansi_text: str):
    mbox = QMessageBox()
    mbox.setIcon(QMessageBox.Icon.Critical)
    mbox.setWindowTitle(title)
    mbox.setTextFormat(Qt.TextFormat.RichText)
    mbox.setText(ansi_to_html(ansi_text))
    mbox.exec()


##############################################################################
# 2) ParameterDialog
##############################################################################
class ParameterDialog(QDialog):
    """Dialog for collecting method parameters (with defaults if available)."""
    def __init__(self, method_name, method_obj, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Enter parameters for {method_name}")
        self.method_obj = method_obj
        self.inputs = {}

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        sig = inspect.signature(method_obj)
        for param_name, param in sig.parameters.items():
            lbl = QLabel(param_name)
            line_edit = QLineEdit()
            if param.default is not inspect.Parameter.empty:
                line_edit.setText(repr(param.default))
            form_layout.addRow(lbl, line_edit)
            self.inputs[param_name] = line_edit

        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_parameters(self):
        """Return {param: value}, or None if parse error."""
        param_dict = {}
        for name, edit in self.inputs.items():
            txt = edit.text().strip()
            if txt:
                try:
                    val = eval(txt)
                    param_dict[name] = val
                except Exception as e:
                    show_colored_error("Parameter Error",
                        f"{RED}Could not parse '{name}' with value '{txt}':{RESET}\n{e}")
                    return None
            else:
                param_dict[name] = None
        return param_dict


##############################################################################
# 3) ReturnedObjectWindow
##############################################################################
class ReturnedObjectWindow(QMainWindow):
    """
    Window for objects returned by method calls. Lists public methods,
    and if calling them returns another object with public methods,
    we open more windows. Also calls parent's on_refresh() if needed.
    """
    def __init__(self, returned_obj, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Methods of Returned Object: {type(returned_obj).__name__}")
        self.returned_obj = returned_obj
        self._parent = parent

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(QLabel("Methods (excluding '_'):"))
        self.method_list = QListWidget()
        layout.addWidget(self.method_list)
        self.method_list.itemClicked.connect(self.on_method_clicked)

        self.populate_methods()

    def populate_methods(self):
        """Populate method_list with public methods."""
        all_methods = dir(self.returned_obj)
        public_methods = []
        for m in all_methods:
            if m.startswith("_"):
                continue
            try:
                attr = getattr(self.returned_obj, m)
                if callable(attr):
                    public_methods.append(m)
            except AttributeError:
                # some attributes might fail on getattr
                continue

        if not public_methods:
            self.method_list.addItem("No recognized public methods")
        else:
            for pm in public_methods:
                self.method_list.addItem(pm)

    def on_method_clicked(self, list_item):
        method_name = list_item.text()
        if method_name.startswith("No recognized"):
            return
        method_obj = getattr(self.returned_obj, method_name, None)
        if not callable(method_obj):
            return

        dlg = ParameterDialog(method_name, method_obj, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            params = dlg.get_parameters()
            if params is None:
                return
            try:
                result = method_obj(**params)
                if result is not None:
                    show_colored_info("Method Called", f"Result:\n{result}")

                # Check if the result has public methods
                if hasattr(result, "__class__"):
                    public_methods_found = False
                    for m in dir(result):
                        if m.startswith("_"):
                            continue
                        try:
                            attr = getattr(result, m)
                            if callable(attr):
                                public_methods_found = True
                                break
                        except AttributeError:
                            continue
                    if public_methods_found:
                        new_win = ReturnedObjectWindow(result, self)
                        new_win.show()
                        new_win.raise_()
                        new_win.activateWindow()

                # After calling a method, refresh parent's tree
                if self._parent:
                    self._parent.on_refresh()

            except Exception as e:
                show_colored_error("Error",
                    f"{RED}Calling method failed:{RESET}\n{e}")


##############################################################################
# 4) DatasetViewer
##############################################################################
class DatasetViewer(QMainWindow):
    """Standalone viewer for HDF5 datasets, with ND slicing > 2D => 2D slice."""
    def __init__(self, filepath, dataset_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"View Dataset: {dataset_path}")

        # Attempt to open file & dataset
        try:
            self.file = h5py.File(filepath, "r")
            self.dataset = self.file[dataset_path]
        except Exception as e:
            show_colored_error("Error",
                f"{RED}Failed to open dataset '{dataset_path}':{RESET}\n{e}")
            self.file = None
            self.dataset = None
            return

        self.shape = self.dataset.shape
        self.ndim = len(self.shape)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.table = QTableWidget()
        layout.addWidget(self.table)

        self.slice_indices = [0]*(self.ndim-2)
        if self.ndim > 2:
            slice_layout = QHBoxLayout()
            self.slice_edits = []
            for i in range(self.ndim - 2):
                lbl = QLabel(f"Dim {i} (0..{self.shape[i]-1}):")
                slice_layout.addWidget(lbl)
                edit = QLineEdit("0")
                edit.editingFinished.connect(self.update_slices)
                slice_layout.addWidget(edit)
                self.slice_edits.append(edit)
            layout.addLayout(slice_layout)

        self.populate_table()
        self.resize(800, 600)

    def closeEvent(self, event):
        if self.file:
            self.file.close()
        super().closeEvent(event)

    def update_slices(self):
        if not self.dataset:
            return
        for i, edit in enumerate(self.slice_edits):
            try:
                val = int(edit.text())
                val = max(0, min(val, self.shape[i]-1))
                self.slice_indices[i] = val
            except:
                self.slice_indices[i] = 0
                edit.setText("0")
        self.populate_table()

    def populate_table(self):
        if not self.dataset:
            return

        self.table.setRowCount(0)
        self.table.setColumnCount(0)

        try:
            # ND=0 => scalar
            if self.ndim == 0:
                data = self.dataset[()]
                self.table.setRowCount(1)
                self.table.setColumnCount(1)
                self.table.setItem(0, 0, QTableWidgetItem(str(data)))
                return

            # ND=1 => 1D
            if self.ndim == 1:
                data = self.dataset[()]
                self.table.setRowCount(len(data))
                self.table.setColumnCount(1)
                for r in range(len(data)):
                    self.table.setItem(r, 0, QTableWidgetItem(str(data[r])))
                return

            # ND=2 => 2D
            if self.ndim == 2:
                data = self.dataset[()]
                rows, cols = data.shape
                self.table.setRowCount(rows)
                self.table.setColumnCount(cols)
                for r in range(rows):
                    for c in range(cols):
                        self.table.setItem(r, c, QTableWidgetItem(str(data[r,c])))
                return

            # ND>2 => build slicing
            slicing = []
            for i, idx in enumerate(self.slice_indices):
                slicing.append(idx)
            slicing.append(slice(None))
            slicing.append(slice(None))
            data_2d = self.dataset[tuple(slicing)]
            if data_2d.ndim != 2:
                show_colored_warning("Warning",
                    f"{RED}After slicing, data has shape {data_2d.shape}, not 2D.{RESET}")
                return

            rows, cols = data_2d.shape
            self.table.setRowCount(rows)
            self.table.setColumnCount(cols)
            for r in range(rows):
                for c in range(cols):
                    self.table.setItem(r, c, QTableWidgetItem(str(data_2d[r, c])))

        except Exception as e:
            show_colored_error("Error",
                f"{RED}Failed to load dataset slice:{RESET}\n{e}")


##############################################################################
# 5) MainWindow
##############################################################################
class MainWindow(QMainWindow):
    """
    Main application window:
      - Open & refresh HDF5 file
      - Tree for main items
      - Middle: method list (reflection on a class from type_registry)
      - Right: attributes, sub-items, sub-sub-items
      - On method invocation => auto-refresh
      - If method returns another object => open ReturnedObjectWindow
      - Use colored message boxes everywhere
      - Remember user selection before refresh, restore after
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDF5 Viewer Refactored - Remember Selection, Less Duplication")
        self.resize(1200, 700)

        splitter = QSplitter(self)
        self.setCentralWidget(splitter)

        # Left: tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type"])
        self.tree.itemClicked.connect(self.on_tree_item_clicked)
        splitter.addWidget(self.tree)

        # Middle: method list
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        splitter.addWidget(middle_widget)
        middle_layout.addWidget(QLabel("Methods (Reflected from class):"))
        self.method_list = QListWidget()
        self.method_list.itemClicked.connect(self.on_method_clicked)
        middle_layout.addWidget(self.method_list)

        # Right: attributes, sub-items, sub-sub-items
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        splitter.addWidget(right_widget)

        right_layout.addWidget(QLabel("Main Item Attributes:"))
        self.main_attr_table = QTableWidget()
        self.main_attr_table.setColumnCount(2)
        self.main_attr_table.setHorizontalHeaderLabels(["Attribute", "Value"])
        right_layout.addWidget(self.main_attr_table)

        right_layout.addWidget(QLabel("Sub-items:"))
        self.subitems_list = QListWidget()
        self.subitems_list.itemClicked.connect(self.on_subitem_clicked)
        right_layout.addWidget(self.subitems_list)

        right_layout.addWidget(QLabel("Sub-sub-items:"))
        self.subsubitems_list = QListWidget()
        self.subsubitems_list.itemClicked.connect(self.on_subsubitem_clicked)
        right_layout.addWidget(self.subsubitems_list)

        right_layout.addWidget(QLabel("Selected Sub/Sub-sub Attributes:"))
        self.sub_attr_table = QTableWidget()
        self.sub_attr_table.setColumnCount(2)
        self.sub_attr_table.setHorizontalHeaderLabels(["Attribute", "Value"])
        right_layout.addWidget(self.sub_attr_table)

        # "View Dataset" button
        self.view_ds_btn = QPushButton("View Dataset")
        self.view_ds_btn.setEnabled(False)
        self.view_ds_btn.clicked.connect(self.on_view_dataset)
        right_layout.addWidget(self.view_ds_btn)

        # Menus
        file_menu = self.menuBar().addMenu("&File")
        open_action = file_menu.addAction("Open HDF5")
        open_action.triggered.connect(self.on_open_file)

        refresh_menu = self.menuBar().addMenu("&Refresh")
        refresh_action = refresh_menu.addAction("Refresh Tree")
        refresh_action.triggered.connect(self.on_refresh)

        # Internal
        self.current_filepath = None
        self.object_map = {}  # path => dict with type_str, reflection_class, etc.

        # We'll store selection paths so we can re-select after refresh:
        self.selected_main_item_path = None
        self.selected_sub_item_path = None
        self.selected_subsub_item_path = None

    ############################################################################
    # File / Refresh
    ############################################################################
    def on_open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open HDF5 File", filter="HDF5 Files (*.h5 *.hdf5 *.slt)"
        )
        if not filepath:
            return
        self.current_filepath = filepath
        self.read_file_structure()

    def on_refresh(self):
        """Remember current selection, reload file, reselect."""
        if not self.current_filepath:
            show_colored_error("Error",
                f"{RED}No file is currently open. Please open a file first.{RESET}")
            return

        # Store current selections
        self.remember_selection()

        # Re-read
        self.read_file_structure()

        # Attempt to restore selections
        self.restore_selection()

    def remember_selection(self):
        """Store the user’s current selection paths for main, sub, sub-sub."""
        # Main tree
        current_tree_item = self.tree.currentItem()
        if current_tree_item:
            self.selected_main_item_path = current_tree_item.text(0)
        else:
            self.selected_main_item_path = None

        # Subitems
        sub_current = self.subitems_list.currentItem()
        if sub_current:
            self.selected_sub_item_path = sub_current.data(Qt.ItemDataRole.UserRole)
        else:
            self.selected_sub_item_path = None

        # Sub-sub
        subsub_current = self.subsubitems_list.currentItem()
        if subsub_current:
            self.selected_subsub_item_path = subsub_current.data(Qt.ItemDataRole.UserRole)
        else:
            self.selected_subsub_item_path = None

    def restore_selection(self):
        """Try to reselect main item, sub item, sub-sub item after refresh."""
        # 1) Reselect main item
        if self.selected_main_item_path is not None:
            item = self.find_tree_item_by_path(self.selected_main_item_path)
            if item:
                self.tree.setCurrentItem(item)
                self.on_tree_item_clicked(item, 0)

        # 2) If we reselected main item, we can restore sub item
        # But we only do that if it’s a group
        if self.selected_sub_item_path:
            # attempt to find sub item in subitems_list
            all_items = (self.subitems_list.item(i) for i in range(self.subitems_list.count()))
            for it in all_items:
                path_val = it.data(Qt.ItemDataRole.UserRole)
                if path_val == self.selected_sub_item_path:
                    self.subitems_list.setCurrentItem(it)
                    self.on_subitem_clicked(it)
                    break

        # 3) sub-sub
        if self.selected_subsub_item_path:
            all_subsub = (self.subsubitems_list.item(i) for i in range(self.subsubitems_list.count()))
            for it2 in all_subsub:
                path_val = it2.data(Qt.ItemDataRole.UserRole)
                if path_val == self.selected_subsub_item_path:
                    self.subsubitems_list.setCurrentItem(it2)
                    self.on_subsubitem_clicked(it2)
                    break

    def find_tree_item_by_path(self, path_str):
        """Traverse top-level items in self.tree to find one whose text(0) == path_str."""
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            child = root.child(i)
            if child.text(0) == path_str:
                return child
        return None

    def read_file_structure(self):
        """Load HDF5 structure into tree and object_map, clearing old data first."""
        self.tree.clear()
        self.object_map.clear()

        try:
            with h5py.File(self.current_filepath, "r") as f:
                for name in f.keys():
                    node = f[name]
                    type_str = node.attrs.get("Type", "Unknown")
                    is_ds = isinstance(node, h5py.Dataset)
                    is_grp = isinstance(node, h5py.Group)

                    reflection_class = type_registry.get(type_str, None)
                    self.object_map[name] = {
                        "type_str": type_str,
                        "reflection_class": reflection_class,
                        "is_dataset": is_ds,
                        "is_group": is_grp
                    }

                    item = QTreeWidgetItem(self.tree)
                    item.setText(0, name)
                    item.setText(1, str(type_str))

        except Exception as e:
            show_colored_error("Error", f"{RED}Failed to open file:{RESET}\n{e}")

    ############################################################################
    # Tree / Sub Items
    ############################################################################
    def on_tree_item_clicked(self, tree_item, column):
        path = tree_item.text(0)
        info = self.object_map.get(path)
        if not info:
            self.view_ds_btn.setEnabled(False)
            return

        # Show methods
        self.populate_methods(path, info)

        # Show attributes
        self.populate_attributes(path, self.main_attr_table)

        # Show sub-items
        self.subitems_list.clear()
        self.subsubitems_list.clear()
        self.sub_attr_table.setRowCount(0)

        if info["is_group"]:
            self.populate_subitems(path, self.subitems_list)

        # Toggle dataset button
        self.view_ds_btn.setEnabled(info["is_dataset"])

    def populate_methods(self, path, info):
        """Reflect on reflection_class to list public methods in method_list."""
        self.method_list.clear()
        reflection_class = info["reflection_class"]
        if reflection_class is None:
            self.method_list.addItem("No recognized public methods")
            return

        try:
            if "_from_slt_file" in dir(reflection_class):
                temp_instance = reflection_class._from_slt_file(SltGroup(self.current_filepath, path))
            else:
                temp_instance = reflection_class(SltGroup(self.current_filepath, path))

            # Find public methods
            all_methods = dir(temp_instance)
            public_methods = []
            for m in all_methods:
                if m.startswith("_"):
                    continue
                try:
                    attr = getattr(temp_instance, m)
                    if callable(attr):
                        public_methods.append(m)
                except AttributeError:
                    # skip
                    continue

            if not public_methods:
                self.method_list.addItem("No recognized public methods")
            else:
                for pm in public_methods:
                    self.method_list.addItem(pm)

        except Exception as exc:
            show_colored_error("Error",
                f"{RED}Failed to reflect on class '{reflection_class}':{RESET}\n{exc}")
            self.method_list.addItem("No recognized public methods")

    def populate_attributes(self, hdf5_path, table):
        """Fill table with attributes of hdf5_path."""
        table.setRowCount(0)
        if not self.current_filepath:
            return
        try:
            with h5py.File(self.current_filepath, "r") as f:
                obj = f[hdf5_path]
                attr_names = list(obj.attrs.keys())
                table.setRowCount(len(attr_names))
                for row, a in enumerate(attr_names):
                    table.setItem(row, 0, QTableWidgetItem(a))
                    table.setItem(row, 1, QTableWidgetItem(str(obj.attrs[a])))
                table.resizeColumnsToContents()
        except Exception as e:
            show_colored_error("Error",
                f"{RED}Reading attributes failed:{RESET}\n{e}")

    def populate_subitems(self, hdf5_path, list_widget):
        """Fill a list_widget with immediate children of hdf5_path (groups/datasets)."""
        list_widget.clear()
        if not self.current_filepath:
            return
        try:
            with h5py.File(self.current_filepath, "r") as f:
                grp = f[hdf5_path]
                for key in grp.keys():
                    if hdf5_path == "/":
                        full_path = f"/{key}"
                    else:
                        full_path = f"{hdf5_path}/{key}"
                    lw_item = QListWidgetItem(key)
                    lw_item.setData(Qt.ItemDataRole.UserRole, full_path)
                    list_widget.addItem(lw_item)
        except Exception as e:
            show_colored_error("Error",
                f"{RED}Listing sub-items failed:{RESET}\n{e}")

    def on_subitem_clicked(self, lw_item):
        """When user picks a sub-item => show its attributes, sub-sub, etc."""
        self.subsubitems_list.clear()
        self.sub_attr_table.setRowCount(0)

        full_path = lw_item.data(Qt.ItemDataRole.UserRole)
        if not full_path:
            self.view_ds_btn.setEnabled(False)
            return

        self.populate_attributes(full_path, self.sub_attr_table)

        try:
            with h5py.File(self.current_filepath, "r") as f:
                node = f[full_path]
                is_ds = isinstance(node, h5py.Dataset)
                is_grp = isinstance(node, h5py.Group)
                self.view_ds_btn.setEnabled(is_ds)

                if is_grp:
                    self.populate_subitems(full_path, self.subsubitems_list)
        except Exception as e:
            show_colored_error("Error",
                f"{RED}Sub-item click error:{RESET}\n{e}")

    def on_subsubitem_clicked(self, lw_item):
        """Show attributes for sub-sub-item, toggle view dataset if it's a dataset."""
        full_path = lw_item.data(Qt.ItemDataRole.UserRole)
        self.populate_attributes(full_path, self.sub_attr_table)
        try:
            with h5py.File(self.current_filepath, "r") as f:
                node = f[full_path]
                self.view_ds_btn.setEnabled(isinstance(node, h5py.Dataset))
        except Exception as e:
            show_colored_error("Error",
                f"{RED}Sub-sub-item click error:{RESET}\n{e}")

    ############################################################################
    # Methods
    ############################################################################
    def on_method_clicked(self, list_item):
        """User clicked a method in the method list => open param dialog => call method."""
        method_name = list_item.text()
        if method_name.startswith("No recognized"):
            return

        current_tree_item = self.tree.currentItem()
        if not current_tree_item:
            return

        hdf5_path = current_tree_item.text(0)
        info = self.object_map.get(hdf5_path)
        if not info:
            return

        reflection_class = info["reflection_class"]
        if reflection_class is None:
            return

        # Make a temp instance to reflect on the method signature
        try:
            if "_from_slt_file" in dir(reflection_class):
                temp_instance = reflection_class._from_slt_file(SltGroup(self.current_filepath, hdf5_path))
            else:
                temp_instance = reflection_class(SltGroup(self.current_filepath, hdf5_path))

            method_obj = getattr(temp_instance, method_name, None)
            if not callable(method_obj):
                return
        except Exception as e:
            show_colored_error("Error",
                f"{RED}Failed to reflect method '{method_name}':{RESET}\n{e}")
            return

        # Param dialog
        dlg = ParameterDialog(method_name, method_obj, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            params = dlg.get_parameters()
            if params is None:
                return
            # Actually call the method on SltGroup
            try:
                slt_g = SltGroup(self.current_filepath, hdf5_path)
                call_method = getattr(slt_g, method_name, None)
                if not callable(call_method):
                    raise AttributeError(f"SltGroup has no method '{method_name}'")

                result = call_method(**params)
                if result is not None:
                    show_colored_info("Method Called", f"Result:\n{result}")

                # If result has public methods => open ReturnedObjectWindow
                if hasattr(result, "__class__"):
                    public_methods_found = False
                    for m in dir(result):
                        if m.startswith("_"):
                            continue
                        try:
                            attr = getattr(result, m)
                            if callable(attr):
                                public_methods_found = True
                                break
                        except AttributeError:
                            continue
                    if public_methods_found:
                        new_win = ReturnedObjectWindow(result, self)
                        new_win.show()
                        new_win.raise_()
                        new_win.activateWindow()

                # After method => refresh
                self.on_refresh()

            except Exception as e:
                show_colored_error("Error",
                    f"{RED}Failed to call method on SltGroup:{RESET}\n{e}")

    ############################################################################
    # Datasets
    ############################################################################
    def on_view_dataset(self):
        """Open DatasetViewer for the selected main/sub/sub-sub item."""
        subsub_item = self.subsubitems_list.currentItem()
        if subsub_item:
            hdf5_path = subsub_item.data(Qt.ItemDataRole.UserRole)
        else:
            sub_item = self.subitems_list.currentItem()
            if sub_item:
                hdf5_path = sub_item.data(Qt.ItemDataRole.UserRole)
            else:
                tree_item = self.tree.currentItem()
                if not tree_item:
                    return
                hdf5_path = tree_item.text(0)

        try:
            viewer = DatasetViewer(self.current_filepath, hdf5_path, self)
            viewer.show()
            viewer.raise_()
            viewer.activateWindow()
        except Exception as e:
            show_colored_error("Error",
                f"{RED}Failed to open dataset viewer:{RESET}\n{e}")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
