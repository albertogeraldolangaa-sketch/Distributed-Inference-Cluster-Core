

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

let win = null;
function createWindow() {
  win = new BrowserWindow({
    width: 1600,
    height: 1000,
    backgroundColor: '#0b0f14',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    }
  });
  win.loadURL(process.env.CLUSTER_DASHBOARD_URL || 'http://127.0.0.1:8080/ui');
}
app.whenReady().then(createWindow);
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
