# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['FEMsolver.py'],
             pathex=['C:\\Users\jpinn_000\Google Drive\Mestrado João\FEM simulator\FEM solver'],
             binaries=[],
             datas=[],
             hiddenimports=['vtkmodules', 'vtkmodules.all', 'vtkmodules.util.numpy_support', 'vtkmodules.numpy_interface', 'vtkmodules.numpy_interface.dataset_adapter','vtkmodules.qt', 'vttmodules.util','vttmodules.vtkCommonCore','vttmodules.vtkCommonKitPython','vtkmodules.qt.QVTKRenderWindowInteractor','pyvista','pyvistaqt'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='JFlow',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
