
const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('clusterBridge', {
  ping: () => ipcRenderer.invoke('ping'),
});
