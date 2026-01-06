"""
Professional logging window with categorized output and search/filter capabilities
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import os


class LoggingWindow(tk.Toplevel):
    """
    Non-modal logging window for displaying categorized system logs
    Features: search, filter, export, auto-scroll
    """

    def __init__(self, parent, log_manager):
        super().__init__(parent)
        self.log_manager = log_manager
        self.parent = parent

        # Configuration
        self.title("Blackjack AI - System Logs")
        self.geometry("1200x700")
        self.configure(bg='#f8f9fa')

        # Non-modal - user can interact with both windows
        # self.grab_set()  # Intentionally commented - we want non-modal

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Color scheme matching modern_gui.py
        self.colors = {
            'bg_primary': '#f8f9fa',
            'bg_secondary': '#ffffff',
            'accent': '#0d6efd',
            'success': '#198754',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'text_primary': '#212529',
            'text_secondary': '#6c757d',
            'border': '#dee2e6',
        }

        # Log category colors
        self.log_colors = {
            'MODEL_LOADING': self.colors['success'],
            'CARD_COUNTER': self.colors['accent'],
            'PERFORMANCE': self.colors['warning'],
            'ERROR': self.colors['danger'],
            'STARTUP': self.colors['success'],
            'CACHE': self.colors['text_secondary'],
            'UI_WARNING': self.colors['warning'],
            'GENERAL': self.colors['text_primary'],
        }

        # UI state variables
        self.auto_scroll = tk.BooleanVar(value=True)
        self.search_var = tk.StringVar()
        self.filter_var = tk.StringVar(value="All")

        # Pending logs for batched updates
        self._pending_logs = []
        self._update_scheduled = False

        # Build UI
        self.build_header()
        self.build_controls()
        self.build_log_display()

        # Register callback with LogManager
        if self.log_manager:
            self.log_manager.register_callback(self.append_log)

        # Load existing logs
        self.after(100, self.load_existing_logs)

    def build_header(self):
        """Build professional header"""
        header_frame = tk.Frame(self, bg=self.colors['bg_secondary'], height=60)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        title_label = tk.Label(header_frame, text="System Logs",
                               font=('Inter', 16, 'bold'),
                               bg=self.colors['bg_secondary'],
                               fg=self.colors['text_primary'])
        title_label.pack(pady=18)

    def build_controls(self):
        """Build control panel with search, filter, and action buttons"""
        control_frame = tk.Frame(self, bg=self.colors['bg_secondary'], height=100)
        control_frame.pack(fill=tk.X, padx=20, pady=10)
        control_frame.pack_propagate(False)

        # First row: Search and Filter
        row1 = tk.Frame(control_frame, bg=self.colors['bg_secondary'])
        row1.pack(fill=tk.X, pady=(0, 8))

        # Search
        search_label = tk.Label(row1, text="🔍 Search:",
                               font=('Inter', 10),
                               bg=self.colors['bg_secondary'],
                               fg=self.colors['text_primary'])
        search_label.pack(side=tk.LEFT, padx=(0, 8))

        search_entry = tk.Entry(row1, textvariable=self.search_var,
                               font=('Inter', 10),
                               width=30)
        search_entry.pack(side=tk.LEFT, padx=(0, 20))

        # Bind search to filter
        self.search_var.trace_add('write', lambda *args: self.filter_logs())

        # Category filter
        filter_label = tk.Label(row1, text="📁 Filter:",
                               font=('Inter', 10),
                               bg=self.colors['bg_secondary'],
                               fg=self.colors['text_primary'])
        filter_label.pack(side=tk.LEFT, padx=(0, 8))

        filter_combo = ttk.Combobox(row1, textvariable=self.filter_var,
                                    state="readonly",
                                    font=('Inter', 10),
                                    width=20)
        filter_combo['values'] = [
            "All",
            "Performance Warnings",
            "Card Counter Updates",
            "Model Loading",
            "Errors Only",
            "Startup",
            "Cache Operations"
        ]
        filter_combo.pack(side=tk.LEFT)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self.filter_logs())

        # Second row: Buttons and Auto-scroll
        row2 = tk.Frame(control_frame, bg=self.colors['bg_secondary'])
        row2.pack(fill=tk.X)

        # Clear button
        clear_btn = tk.Button(row2, text="Clear Logs",
                             command=self.clear_logs,
                             bg=self.colors['warning'],
                             fg=self.colors['text_primary'],
                             font=('Inter', 10),
                             relief='flat',
                             cursor='hand2',
                             padx=15, pady=8)
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Export button
        export_btn = tk.Button(row2, text="Export to File",
                              command=self.export_logs,
                              bg=self.colors['accent'],
                              fg='white',
                              font=('Inter', 10),
                              relief='flat',
                              cursor='hand2',
                              padx=15, pady=8)
        export_btn.pack(side=tk.LEFT, padx=(0, 20))

        # Auto-scroll checkbox
        auto_scroll_check = tk.Checkbutton(row2, text="Auto-scroll to latest",
                                          variable=self.auto_scroll,
                                          font=('Inter', 10),
                                          bg=self.colors['bg_secondary'],
                                          fg=self.colors['text_primary'],
                                          selectcolor=self.colors['bg_primary'],
                                          activebackground=self.colors['bg_secondary'])
        auto_scroll_check.pack(side=tk.LEFT)

        # Log count label
        self.count_label = tk.Label(row2, text="Logs: 0",
                                   font=('Inter', 10),
                                   bg=self.colors['bg_secondary'],
                                   fg=self.colors['text_secondary'])
        self.count_label.pack(side=tk.RIGHT, padx=10)

    def build_log_display(self):
        """Build scrollable log display area"""
        display_frame = tk.Frame(self, bg=self.colors['bg_primary'])
        display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Create Text widget with scrollbar
        scrollbar = tk.Scrollbar(display_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_widget = tk.Text(display_frame,
                                   wrap=tk.WORD,
                                   font=('Consolas', 9),
                                   bg=self.colors['bg_secondary'],
                                   fg=self.colors['text_primary'],
                                   yscrollcommand=scrollbar.set,
                                   state='disabled',
                                   relief='flat',
                                   borderwidth=1,
                                   highlightbackground=self.colors['border'],
                                   highlightthickness=1)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.text_widget.yview)

        # Configure color tags for each category
        for category_name, color in self.log_colors.items():
            self.text_widget.tag_config(f"category_{category_name}",
                                       foreground=color,
                                       font=('Consolas', 9))

        # Highlight tag for search results
        self.text_widget.tag_config("highlight",
                                   background='#ffeb3b',
                                   foreground=self.colors['text_primary'])

        # Detect manual scrolling to disable auto-scroll
        self.text_widget.bind('<MouseWheel>', self._on_manual_scroll)
        self.text_widget.bind('<Button-4>', self._on_manual_scroll)
        self.text_widget.bind('<Button-5>', self._on_manual_scroll)

    def _on_manual_scroll(self, event):
        """Disable auto-scroll when user manually scrolls"""
        # Check if scrolled away from bottom
        if self.text_widget.yview()[1] < 0.99:
            self.auto_scroll.set(False)

    def append_log(self, log_entry):
        """
        Add a new log entry (called by LogManager callback)
        Uses batching to prevent UI freeze
        """
        # Check if window still exists before scheduling updates
        if not self.winfo_exists():
            return

        self._pending_logs.append(log_entry)

        # Schedule batch processing
        if not self._update_scheduled:
            self._update_scheduled = True
            try:
                self.after(100, self._process_pending_logs)
            except tk.TclError:
                # Window was destroyed - ignore
                self._update_scheduled = False

    def _process_pending_logs(self):
        """Process batched log updates"""
        # Check if window still exists before processing
        if not self.winfo_exists():
            self._update_scheduled = False
            self._pending_logs.clear()
            return

        if not self._pending_logs:
            self._update_scheduled = False
            return

        try:
            self.text_widget.config(state='normal')

            for entry in self._pending_logs:
                # Apply filters
                if not self._matches_filters(entry):
                    continue

                # Format log entry
                timestamp_str = entry.timestamp.strftime("%H:%M:%S")
                category_str = f"[{entry.category.value:15}]"
                formatted = f"[{timestamp_str}] {category_str} {entry.message}\n"

                # Insert with color tag
                tag_name = f"category_{entry.category.name}"
                self.text_widget.insert(tk.END, formatted, tag_name)

            self.text_widget.config(state='disabled')

            # Auto-scroll to bottom if enabled
            if self.auto_scroll.get():
                self.text_widget.see(tk.END)

            # Update count
            self._update_count_label()

        except tk.TclError:
            # Window was destroyed while processing - ignore
            pass
        finally:
            self._pending_logs.clear()
            self._update_scheduled = False

    def _matches_filters(self, entry):
        """Check if log entry matches current search and filter criteria"""
        # Search filter
        search_text = self.search_var.get().lower()
        if search_text and search_text not in entry.message.lower():
            return False

        # Category filter
        filter_value = self.filter_var.get()
        if filter_value == "All":
            return True
        elif filter_value == "Performance Warnings" and entry.category.name != "PERFORMANCE":
            return False
        elif filter_value == "Card Counter Updates" and entry.category.name != "CARD_COUNTER":
            return False
        elif filter_value == "Model Loading" and entry.category.name != "MODEL_LOADING":
            return False
        elif filter_value == "Errors Only" and entry.level != "ERROR":
            return False
        elif filter_value == "Startup" and entry.category.name != "STARTUP":
            return False
        elif filter_value == "Cache Operations" and entry.category.name != "CACHE":
            return False

        return True

    def filter_logs(self):
        """Re-display logs with current filter settings"""
        if not self.log_manager or not self.winfo_exists():
            return

        try:
            # Clear display
            self.text_widget.config(state='normal')
            self.text_widget.delete('1.0', tk.END)

            # Re-add all logs that match filters
            all_logs = self.log_manager.get_all_logs()
            for entry in all_logs:
                if self._matches_filters(entry):
                    timestamp_str = entry.timestamp.strftime("%H:%M:%S")
                    category_str = f"[{entry.category.value:15}]"
                    formatted = f"[{timestamp_str}] {category_str} {entry.message}\n"

                    tag_name = f"category_{entry.category.name}"
                    self.text_widget.insert(tk.END, formatted, tag_name)

            self.text_widget.config(state='disabled')

            # Auto-scroll to bottom
            if self.auto_scroll.get():
                self.text_widget.see(tk.END)

            # Update count
            self._update_count_label()

        except tk.TclError:
            # Window was destroyed - ignore
            pass

    def clear_logs(self):
        """Clear all logs from display and manager"""
        if messagebox.askyesno("Confirm Clear", "Clear all logs? This cannot be undone."):
            if self.log_manager:
                self.log_manager.clear_logs()

            self.text_widget.config(state='normal')
            self.text_widget.delete('1.0', tk.END)
            self.text_widget.config(state='disabled')

            self._update_count_label()

    def export_logs(self):
        """Export logs to a text file"""
        if not self.log_manager:
            return

        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)

        # Default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"blackjack_{timestamp}.log"
        default_path = os.path.join(logs_dir, default_filename)

        # Ask user for save location
        filepath = filedialog.asksaveasfilename(
            title="Export Logs",
            initialdir=logs_dir,
            initialfile=default_filename,
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if not filepath:
            return  # User cancelled

        try:
            # Get all logs
            all_logs = self.log_manager.get_all_logs()

            with open(filepath, 'w', encoding='utf-8', errors='replace') as f:
                f.write(f"Blackjack AI - System Logs Export\n")
                f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Logs: {len(all_logs)}\n")
                f.write("=" * 80 + "\n\n")

                for entry in all_logs:
                    timestamp_str = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp_str}] [{entry.level:7}] [{entry.category.value:15}] {entry.message}\n")

            messagebox.showinfo("Export Successful",
                              f"Logs exported to:\n{filepath}\n\nTotal entries: {len(all_logs)}")

        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export logs:\n{e}")

    def load_existing_logs(self):
        """Load and display all existing logs from LogManager"""
        if not self.log_manager or not self.winfo_exists():
            return

        try:
            all_logs = self.log_manager.get_all_logs()

            self.text_widget.config(state='normal')

            for entry in all_logs:
                if self._matches_filters(entry):
                    timestamp_str = entry.timestamp.strftime("%H:%M:%S")
                    category_str = f"[{entry.category.value:15}]"
                    formatted = f"[{timestamp_str}] {category_str} {entry.message}\n"

                    tag_name = f"category_{entry.category.name}"
                    self.text_widget.insert(tk.END, formatted, tag_name)

            self.text_widget.config(state='disabled')

            # Scroll to bottom
            if self.auto_scroll.get():
                self.text_widget.see(tk.END)

            # Update count
            self._update_count_label()

        except tk.TclError:
            # Window was destroyed - ignore
            pass

    def _update_count_label(self):
        """Update the log count label"""
        if not self.winfo_exists():
            return

        try:
            # Count visible lines
            content = self.text_widget.get('1.0', tk.END)
            line_count = content.count('\n') - 1  # Subtract 1 for trailing newline
            self.count_label.config(text=f"Logs: {line_count}")
        except tk.TclError:
            # Window was destroyed - ignore
            pass

    def on_close(self):
        """Handle window close event"""
        # Unregister callback
        if self.log_manager:
            self.log_manager.unregister_callback(self.append_log)

        # Destroy window
        self.destroy()
