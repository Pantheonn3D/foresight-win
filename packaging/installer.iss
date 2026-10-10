; Inno Setup script. Per-user install, no admin rights needed.
; Build:  iscc /DAppVersion=0.1.0 packaging\installer.iss   (expects dist\Foresight from PyInstaller)

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6F1C2B7E-3A0D-4E8B-9C55-2D7A0F4B91E3}
AppName=Foresight
AppVersion={#AppVersion}
AppPublisher=Foresight for Windows
DefaultDirName={localappdata}\Programs\Foresight
DefaultGroupName=Foresight
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=Foresight-Setup-{#AppVersion}
SetupIconFile=foresight.ico
UninstallDisplayIcon={app}\Foresight.exe
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\Foresight\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\Foresight"; Filename: "{app}\Foresight.exe"
Name: "{autodesktop}\Foresight"; Filename: "{app}\Foresight.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Foresight.exe"; Description: "Start Foresight"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; models, meetings and settings live in %LOCALAPPDATA%\Foresight and are kept on uninstall on purpose;
; the uninstaller only removes the program folder.
Type: filesandordirs; Name: "{app}"
