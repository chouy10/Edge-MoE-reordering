# Lowrank
I tried Leyao’s old version SSMU baseline code and applied the low rank in B C and DELTA. As a result, Apply lowrank in B can improve latency and resources but apply B and C or DELTA at the same time will decrease.

old version SSMU baseline:
<img width="797" height="211" alt="image" src="https://github.com/user-attachments/assets/fcdbfe1c-fa5c-4770-bca7-1dfab4db120f" />

Cahnging Parts:
ssmu1:
Only apply B
<img width="614" height="460" alt="image" src="https://github.com/user-attachments/assets/4a710907-5c67-42c8-ba9a-87d35aa1c14e" />

Lowrank B:
<img width="799" height="292" alt="image" src="https://github.com/user-attachments/assets/909d970c-b551-4f6f-9370-e51ff4a0d738" />

Simulation B:
<img width="958" height="200" alt="image" src="https://github.com/user-attachments/assets/59aedb6e-2a82-4456-acda-0d208bf344e1" />


