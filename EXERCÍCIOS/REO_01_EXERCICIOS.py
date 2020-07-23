########################################################################################################################
# DATA: 02/07/2020
print("REO-1")
print("DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS")
print("PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO")
print("Aluno: Everton da Silva Cardoso")
# E-MAIL: vinicius.carneiro@ufla.br
# GITHUB: vqcarneiro
print("------------------------------------------------------------")
########################################################################################################################

print("REO 01 - LISTA DE EXERCÍCIOS")
print("-------------------------------------------------------------")
# EXERCÍCIO 01:
# a) Declare os valores 43.5,150.30,17,28,35,79,20,99.07,15 como um array numpy.
print("Questão 1A-Declare os valores :\n"
      "43.5,150.30,17,28,35,79,20,99.07,15 como um array numpy.")
print(" ")
import numpy as np
lista= [43.5,150.30,17,28,35,79,20,99.07,15]
print("lista: ", lista)
print("tipo:")
print(type(lista))
vetor = np.array(lista)
print("vetor:", vetor)
print("tipo:")
print(type(vetor))

print("-------------------------------------------------------------")

# b) Obtenha as informações de dimensão, média, máximo, mínimo e variância deste vetor.
print("Questão B - Dimensão, média, máximo, mínimo e variância deste vetor dimensão n° dados e por shape")
dim=len(vetor)
print("Dimensão:", dim )
dim02=vetor.shape
print("dimensão(shape):", dim02)
print("A média da lista é {}".format(np.mean(vetor)))
print("O valor máximo é {}".format(np.max(vetor)))
print("O valor mínimo é {}".format(np.min(vetor)))
print("A variancia da lista é {}".format(np.var(vetor)))

print("-------------------------------------------------------------")
# c) Obtenha um novo vetor em que cada elemento é dado pelo quadrado da diferença entre cada elemento do vetor declarado
# na letra a e o valor da média deste.
print("Questão 1C -Novo vetor em que cada elemento é dado pelo quadrado \n"
      "da diferença entre cada elemento e sua média")
mediav1=np.mean(vetor)
vetor2=((mediav1-vetor)**2)
print("Nova lista com o (diferença de x da média ao quadrado:")
print(vetor2)
print("média da nova lista é {} .".format(np.mean(vetor2)))

print("-------------------------------------------------------------")

# d) Obtenha um novo vetor que contenha todos os valores superiores a 30.
print("Questão 1D - Novo vetor que contenha todos os valores superiores a 30")
vetorma30= vetor[vetor>30]
print("OS Valores maiores que 30 dentro da lista 1 são", vetorma30)

print("-------------------------------------------------------------")
# e) Identifique quais as posições do vetor original possuem valores superiores a 30
print("Questão 1E - Posições do vetor original possuem valores superiores a 30")
posma30=np.where(vetor>30)
print("lista geral:", vetor)
print("posições: ", posma30[0])
print("-------------------------------------------------------------")
# f) Apresente um vetor que contenha os valores da primeira, quinta e última posição.
print("Questão 1F - valores da primeira, quinta e última posição")
vetorpos1=vetor[0]
vetorpos5=vetor[4]
vetorpos9=vetor[-1]
print("lista original:", vetor)
print(" primeira posição:", vetorpos1)
print(" quinta posição:", vetorpos5)
print(" ultima posição:", vetorpos9)
print("-------------------------------------------------------------")
# g) Crie uma estrutura de repetição usando o for para apresentar cada valor e a sua respectiva posição durante as iterações
print("Questão 1G - apresentação de cada valor e a sua respectiva posição.")
for pos, valor in enumerate(vetor):
    print("Na posição {} há o valor {}".format(pos, valor))
print("-------------------------------------------------------------")
# h) Crie uma estrutura de repetição usando o for para fazer a soma dos quadrados de cada valor do vetor.
print("Questão 1H - A soma dos quadrados de cada valor do vetor")

somador=0
it=1
for num in vetor:
        print("iteração {}, numero do vetor {}, numero ao quadrado {}.".format(it, num, num**2))
        somador=(num**2+somador)
        it=it+1
        print("soma dos quadrados totais {}".format(somador))

print("-------------------------------------------------------------")
# i) Crie uma estrutura de repetição usando o while para apresentar todos os valores do vetor
print("Questão 1I- Apresentando todos os valores com while ")

import time
contador=0
while contador !=9:
         print(vetor[0+contador])
         contador = contador + 1
#         time.sleep(0.5)

print("Confirmação da lista:", vetor)
print("-------------------------------------------------------------")

# j) Crie um sequência de valores com mesmo tamanho do vetor original e que inicie em 1 e o passo seja também 1.

print("Questão 1J- sequência de valores com mesmo tamanho do vetor \n"
      "original e que inicie em 1 e o passo seja também 1")
lista1a9= (list(range(1,len(vetor)+1)))
print("Lista:", lista1a9)
# h) Concatene o vetor da letra a com o vetor da letra j.
concatenados=np.concatenate((vetor, lista1a9), axis=0)
print(concatenados)

print("-------------------------------------------------------------")
print("QUESTÃO 2 - REO 1")
print("-------------------------------------------------------------")
########################################################################################################################
########################################################################################################################
########################################################################################################################
import numpy as np
# Exercício 02
#a) Declare a matriz abaixo com a biblioteca numpy.
# 1 3 22
# 2 8 18
# 3 4 22
# 4 1 23
# 5 2 52
# 6 2 18
# 7 2 25
print("Questão 2A - Declarando a matriz como numpy")
print("1 3 22 \n2 8 18 \n3 4 22 \n4 1 23 \n5 2 52 \n6 2 18 \n7 2 25")

matriz= np.array([[1, 3, 22],[2, 8, 18],[ 3, 4, 22 ],[4, 1, 23,],[5, 2, 52],[6, 2, 18, ],[7, 2, 25]])

print(" matriz definida como numpy array:")
print(matriz)
print("-------------------------------------------------------------")

# b) Obtenha o número de linhas e de colunas desta matriz
print("Questão 2B - Dimensão da matriz")
diml, dimc = np.shape(matriz)
print("  -  A matriz possui as seguintes dimensões: {} linhas e {} colunas".format(diml, dimc))
print("-------------------------------------------------------------")

# c) Obtenha as médias das colunas 2 e 3.
print("Questão 2C -  Médias das colunas 2 e 3")
matrizcol2= matriz[:, 1]
matrizcol3=matriz[:, 2]
mediacol2= np.mean(matrizcol2)
mediacol3=np.mean(matrizcol3)
print(" A coluna 2 da matriz possui os valores {}, e sua respectiva média é {}".format(matrizcol2, mediacol2))
print(" A coluna 3 da matriz possui os valores {}, e sua respectiva média é {}".format(matrizcol3, mediacol3))
print("-------------------------------------------------------------")

# d) Obtenha as médias das linhas considerando somente as colunas 2 e 3
print("Questão 2D -  Médias das linhas 2 e 3")
matrizl1= matriz[0,[1,2]]
matrizl2= matriz[1,[1,2]]
matrizl3= matriz[2,[1,2]]
matrizl4= matriz[3,[1,2]]
matrizl5= matriz[4,[1,2]]
matrizl6= matriz[5,[1,2]]
matrizl7= matriz[6,[1,2]]
medial1_23= np.mean(matrizl1)
medial2_23= np.mean(matrizl2)
medial3_23= np.mean(matrizl3)
medial4_23= np.mean(matrizl4)
medial5_23= np.mean(matrizl5)
medial6_23= np.mean(matrizl6)
medial7_23= np.mean(matrizl7)

print(" A média da linha 1, com os valores {}, considerando as colunas 2 e 3 é {},  ".format(matrizl1, medial1_23))
print(" A média da linha 2, com os valores {}, considerando as colunas 2 e 3 é {},  ".format(matrizl2, medial2_23))
print(" A média da linha 3, com os valores {}, considerando as colunas 2 e 3 é {},  ".format(matrizl3, medial3_23))
print(" A média da linha 4, com os valores {}, considerando as colunas 2 e 3 é {},  ".format(matrizl4, medial4_23))
print(" A média da linha 5, com os valores {}, considerando as colunas 2 e 3 é {},  ".format(matrizl5, medial5_23))
print(" A média da linha 6, com os valores {}, considerando as colunas 2 e 3 é {},  ".format(matrizl6, medial6_23))
print(" A média da linha 7, com os valores {}, considerando as colunas 2 e 3 é {},  ".format(matrizl7, medial7_23))


print("-------------------------------------------------------------")
# e) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade inferior a 5.
print("Questão 2E - Considerando que a primeira coluna seja a identificação de genótipos, a \n"
      "segunda nota de severidade de uma doença e a terceira peso de 100 grãos. Obtenha os \n"
      "genótipos que possuem nota de severidade inferior a 5")
severidade= matriz[:,1]
nme5sev= np.where(severidade<5)
gennotame5sev=nme5sev[0]+1
print(" Os genótipos, que possuem notas notas de severidade a 5 são {} .".format(gennotame5sev))

print("-------------------------------------------------------------")

# f) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de peso de 100 grãos superior ou igual a 22.
print("Questão 2F -  Considerando que a primeira coluna seja a identificação de genótipos, \n "
      "a segunda nota de severidade de uma doença ee a terceira peso de 100 grãos. Obtenha os \n"
      "genótipos que possuem nota de peso de 100 grãos superior ou igual a 22")
print(" ")
pgrão=matriz[:,2]
pgramam22= np.where(pgrão>=22)
genpegraomam22=pgramam22[0]+1

print(" Os genótipos que possuem peso de grão maior ou igual a 22 são {} .".format(genpegraomam22))

print("-------------------------------------------------------------")
# g) Considerando que a primeira coluna seja a identificação de genótipos, a segunda nota de severidade de uma doença e
# e a terceira peso de 100 grãos. Obtenha os genótipos que possuem nota de severidade igual ou inferior a 3 e peso de 100
# grãos igual ou superior a 22.
print("Questão 2G - Considerando que a primeira coluna seja a identificação de genótipos,\n"
      " a segunda nota de severidade de uma doença e a terceira peso de 100 grãos. \n "
      "Obtenha os genótipos que possuem nota de severidade igual ou inferior a 3 e peso \n"
      "de 100 grãos igual ou superior a 22.")
print(" ")
notasevmem3= np.where(severidade<=3)
print ("Os genótipos que possuem nota de severidade menor igual a 3 são {}.".format(notasevmem3[0]+1))
print ("Os genótipos que possuem peso de grão igual maior a 22 são {}.".format(pgramam22[0]+1))

it=0
for b in notasevmem3[0]:
        it =it+1
        print("O genótipo {} {}".format(b, pgramam22[0][:5] == b))
print("O genótipos que possuem TRUE em sua linha apresentam ambas caracteristicas")

print("-------------------------------------------------------------")

# h) Crie uma estrutura de repetição com uso do for (loop) para apresentar na tela cada uma das posições da matriz e o seu
#  respectivo valor. Utilize um iterador para mostrar ao usuário quantas vezes está sendo repetido.
#  Apresente a seguinte mensagem a cada iteração "Na linha X e na coluna Y ocorre o valor: Z".
#  Nesta estrutura crie uma lista que armazene os genótipos com peso de 100 grãos igual ou superior a 25
print("Questão 2H - Crie uma estrutura de repetição com uso do for (loop) para apresentar na tela cada \n"
      "uma das posições da matriz e o seu respectivo valor. Utilize um iterador para mostrar\n"
      "ao usuário quantas vezes está sendo repetido. Apresente a seguinte mensagem a cada \n"
      "iteração. Na linha X e na coluna Y ocorre o valor: Z. Nesta estrutura crie uma lista \n" \
       "que armazene os genótipos com peso de 100 grãos igual ou superior a 25")

matrizcol1=matriz[:, 0]
it=0
ib=0
genotipomampg25=[]
print(genotipomampg25)
print(matriz)
for col1, col2, col3 in zip(matrizcol1, matrizcol2, matrizcol3):
    it = it + 1
    print("Na linha {} e na coluna {} ocorre o valor :{}".format(0+ib,0, col1))
    print("Na linha {} e na coluna {} ocorre o valor :{}".format(0+ib,0+1, col2))
    print("Na linha {} e na coluna {} ocorre o valor :{}".format(0+ib,0+2, col3))
    ib = ib + 1
    print("interação n° {}".format(it))
    print(" ")
    if col3 > 24 :
        add = col1
        genotipomampg25.append(add)

print("os genótipos com peso de 100 grãos igual ou superior a 25 são : {}".format(genotipomampg25))

print("-------------------------------------------------------------")

########################################################################################################################
########################################################################################################################
########################################################################################################################

import numpy as np
# EXERCÍCIO 03:
print("QUESTÃO 3 - REO 1")
print("-------------------------------------------------------------")

# a) Crie uma função em um arquivo externo (outro arquivo .py) para calcular a média e a variância amostral um vetor qualquer, baseada em um loop (for).
print("Questão 3A - Crie uma função em um arquivo externo (outro arquivo .py) \n"
      "para calcular a média e a variância amostral um vetor qualquer, \n"
      "baseada em um loop (for).")
print(" ")
vetor3=np.array(list(range(10,101,10)))
print("O vetor Adicionado foi:", vetor3)
import sys

from funções_REO1 import f_media

media3 = f_media(vetor3)
print("Sua média atráves da função f_media (origem: função_REO1.py) É :{}.".format(media3))
from funções_REO1 import f_var_amostral
varamostral=f_var_amostral(vetor3)
print("Sua variancia amostral atráves da função f_varamostral (origem: função_REO1.py) É :{}.".format(varamostral))
print(" ")
print("-------------------------------------------------------------")

# b) Simule três arrays com a biblioteca numpy de 10, 100, e 1000 valores e com distribuição normal com média 100 e variância 2500. Pesquise na documentação do numpy por funções de simulação.
print("Simule três arrays com a biblioteca numpy de 10, 100, e 1000 valores e com\n"
      " distribuição normal com média 100 e variância 2500. \n"
      "Pesquise na documentação do numpy por funções de simulação.")

vetorsimulado10=np.random.normal(100, 50, 10)
vetorsimulado100=np.random.normal(100, 50, 100)
vetorsimulado1000=np.random.normal(100, 50, 1000)
vetorsimulado10000=np.random.normal(100, 50, 10000)
print("-------------------------------------------------------------")

# c) Utilize a função criada na letra a para obter as médias e variâncias dos vetores simulados na letra b.
print(" 3C - Utilize a função criada na letra a para obter as médias e variâncias dos vetores simulados na letra b")
print(" ")
mediavetor10=f_media(vetorsimulado10)
print(" A confirmação da média através da função f_media (origem: função_REO1.py) \n"
      "para o vetor criado com 10 elementos é : {}".format(mediavetor10))
print(" ")
mediavetor100=f_media(vetorsimulado100)
print(" A confirmação da média através da função f_media (origem: função_REO1.py) \n"
      "para o vetor criado com 100 elementos é : {}".format(mediavetor100))
print(" ")
mediavetor1000=f_media(vetorsimulado1000)
print(" A confirmação da média através da função f_media (origem: função_REO1.py) \n"
      "para o vetor criado com 1000 elementos é : {}".format(mediavetor1000))
print("-------------------------------------------------------------")
# d) Crie histogramas com a biblioteca matplotlib dos vetores simulados com valores de 10, 100, 1000 e 100000.
print(" QUESTÃO 3D - Histogramas com a biblioteca matplotlib dos vetores simulados \n"
      "com valores de 10, 100, 1000 e 100000.")

from matplotlib import pyplot as plt





font = {'size': 12}
import matplotlib.pyplot as plt


vetorsimulado10000=np.random.normal(100, 50, 10000)

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()
ax0.hist(vetorsimulado10, color="tab:red")
ax0.set_title('10 genotipos', fontdict=font, color="tab:red" )
ax1.hist(vetorsimulado100, color="tab:orange")
ax1.set_title('100 genotipos', fontdict=font, color="tab:orange" )
ax2.hist(vetorsimulado1000, color="tab:green")
ax2.set_title('1000 genotipos', fontdict=font, color="tab:green")
ax3.hist(vetorsimulado10000, color="tab:blue")
ax3.set_title('10000 genotipos', fontdict=font,color="tab:blue" )
fig.tight_layout()
plt.show()

print(' ')
print('-='*100)
print(' ')
print("-------------------------------------------------------------")

########################################################################################################################
########################################################################################################################
########################################################################################################################
# EXERCÍCIO 04:
# a) O arquivo dados.txt contem a avaliação de genótipos (primeira coluna) em repetições (segunda coluna) quanto a quatro
print("QUESTÃO 4A -O arquivo dados.txt contem a avaliação de genótipos (primeira coluna) \n"
      "em repetições (segunda coluna) quanto a CINCO variáveis (terceira coluna em diante).\n"
      " Portanto, carregue o arquivo dados.txt com a biblioteca numpy, apresente os dados e \n"
      "obtenha as informaçõesde dimensão desta matriz.")
import numpy as np
dados = np.loadtxt('dados.txt')
print(" Dados carregados :")
print(" ")
print(dados)
print(" ")
nldados, ncdados= np.shape(dados)
print(" Os dados carregados possuem {} linhas, e {} colunas .".format(nldados, ncdados))
print("-------------------------------------------------------------")
# b) Pesquise sobre as funções np.unique e np.where da biblioteca numpy
print("QUESTÃO 4B - funções np.unique e np.where da biblioteca numpy")

'''help(np.unique)
print(" ")
help(np.where)
'''
print("-------------------------------------------------------------")
# c) Obtenha de forma automática os genótipos e quantas repetições foram avaliadas
print("QUESTÃO 4C - Obtenha de forma automática os genótipos e quantas repetições foram avaliadas")
print("")
numgenotipos=np.unique(dados[:,0])
numrep=np.unique(dados[:, 1])
print("O número de genótipos avaliados é : {}, com {} Repetições".format(numgenotipos [-1], numrep [-1]))
print("-------------------------------------------------------------")

# d) Apresente uma matriz contendo somente as colunas 1, 2 e 4
print(" QUESTÃO 4D - Matriz contendo somente as colunas 1, 2 e 4")
dadoscol124= dados[:,[0,1,3]]
print(dadoscol124 )
print("-------------------------------------------------------------")
genotipos=np.unique(dados[0:30,0:1], axis=0)
# e) Obtenha uma matriz que contenha o máximo, o mínimo, a média e a variância de cada genótipo para a variavel da coluna 4. Salve esta matriz em bloco de notas.
print(" QUESTÃO 4E - Obtenha uma matriz que contenha o máximo, o mínimo, a média e a variância de cada \n"
      "genótipo para a variavel da coluna 4. Salve esta matriz em bloco de notas.")

mingen= np.zeros((10,1))
maxgen= np.zeros((10,1))
meangen=np.zeros((10,1))
vargen=np.zeros((10,1))


it=0
for el in np.arange(0,nldados,3):
    maxgen[it,0] = np.max(dadoscol124[el:el + 3, 2], axis=0)
    meangen[it,0] = np.mean(dadoscol124[el:el + 3, 2], axis=0)
    vargen[it,0] = np.var(dadoscol124[el:el + 3, 2], axis=0)
    it += 1

#print('-       Genótipos / Min / Max / Média / Variância       -')
matriz_final = np.concatenate((genotipos, mingen,maxgen, meangen, vargen), axis=1 )
print(matriz_final)
fmt='%1.0f', '%10.2f', '%10.2f', '%10.2f', '%10.2f'
np.savetxt("tabelagerada_ex3.txt", matriz_final,fmt=fmt, delimiter=" ", header= "Genótipos, Min, Max, Média, Variância } da variavel da col 4")
print("-------------------------------------------------------------")
# f) Obtenha os genótipos que possuem média (médias das repetições) igual ou superior a 500 da matriz gerada na letra anterior.
print("4F - Obtenha os genótipos que possuem média (médias das repetições) igual ou superior a \n"
      "500 da matriz gerada na letra anterior.")
medmam500=[]
for g, md in zip(matriz_final[:,0], matriz_final[:,3]):
    if md > 499:
        apd= g
        medmam500.append(apd)
print(" os genótipos com Valores média >= 500 são {}".format(medmam500))
print("-------------------------------------------------------------")

print("4G - Apresente os seguintes gráficos:")
print(  " -   Médias dos genótipos para cada variável. Utilizar o comando plt.subplot para \n"
            "mostrar mais de um grafico por figura")
# g) Apresente os seguintes graficos:
#    - Médias dos genótipos para cada variável. Utilizar o comando plt.subplot para mostrar mais de um grafico por figura

from matplotlib import pyplot as plt

media1 = np.zeros((10,1))
media2 = np.zeros((10,1))
media3 = np.zeros((10,1))
media4 = np.zeros((10,1))
media5 = np.zeros((10,1))
it = 0
for i in np.arange(0,30,3):
    media1[it,0] = np.mean(dados[i:i + 3, 2], axis=0)
    media2[it,0] = np.mean(dados[i:i + 3, 3], axis=0)
    media3[it,0] = np.mean(dados[i:i + 3, 4], axis=0)
    media4[it,0] = np.mean(dados[i:i + 3, 5], axis=0)
    media5[it,0] = np.mean(dados[i:i + 3, 6], axis=0)
    it = it + 1

genotipos=np.unique(dados[0:30,0:1], axis=0)

md_geral = np.concatenate((genotipos,media1,media2,media3,media4,media5),axis=1)
print(md_geral)
plt.style.use('ggplot')
plt.figure('Grafíco das variáveis')
font = {'family': 'serif',
        'color':  'gray',
        'weight': 'normal',
        'size': 8,
        }
font1 = {'family': 'serif',
        'color':  'gray',
        'weight': 'normal',
        'size': 9,
        }


plt.subplot(2,3,1)
plt.bar(md_geral[:,0],md_geral[:,1])
plt.title('Var. 1 - col 3', fontdict=font)
plt.xticks(md_geral[:,0])
plt.xlabel('Genótipos', fontdict=font)
plt.ylabel('Média', fontdict=font)

plt.subplot(2,3,2)
plt.bar(md_geral[:,0],md_geral[:,2])
plt.title('Var. 2 - col 4', fontdict=font)
plt.xticks(md_geral[:,0])
plt.xlabel('Genótipos', fontdict=font)
plt.ylabel('Média', fontdict=font)

plt.subplot(2,3,3)
plt.bar(md_geral[:,0],md_geral[:,3])
plt.title('Var. 3 - col 5', fontdict=font)
plt.xticks(md_geral[:,0])
plt.xlabel('Genótipos', fontdict=font)
plt.ylabel('Média', fontdict=font)

plt.subplot(2,3,4)
plt.bar(md_geral[:,0],md_geral[:,4])
plt.text(5, 430, 'Var. 4 - col 6', fontdict=font1)
plt.xticks(md_geral[:,0])
plt.xlabel('Genótipos', fontdict=font)
plt.ylabel('Média', fontdict=font)

plt.subplot(2,3,5)
plt.bar(md_geral[:,0],md_geral[:,5])
plt.text(5, 630, 'Var. 5 - col 7', fontdict=font1)
plt.xticks(md_geral[:,0])
plt.xlabel('Genótipos', fontdict=font)
plt.ylabel('Média', fontdict=font)
plt.show()

print(' ')

print("  - Dispersão 2D da médias dos genótipos (Utilizar as três primeiras variáveis\n"
        " No eixo X uma variável e no eixo Y outra")

plt.style.use('ggplot')
fig = plt.figure('Scatter plot 2D')
plt.subplot(2,2,1)

c = ['yellow','red','green','black','pink','black','orange','cyan','slategray','gray']

for i in np.arange(0,10,1):
    plt.scatter(md_geral[i,1], md_geral[i,2],label = md_geral[i,0],c = c[i])

plt.xlabel('Var 1', fontdict=font1)
plt.ylabel('Var 2', fontdict=font1)
plt.subplot(2,2,2)

for i in np.arange(0,10,1):
    plt.scatter(md_geral[i,1], md_geral[i,3], label = md_geral[i,0], c=c[i])

plt.xlabel('Var 1', fontdict=font1)
plt.ylabel('Var 3', fontdict=font1)
plt.subplot(2,2,3)

for i in np.arange(0,10,1):
    plt.scatter(md_geral[i,1], md_geral[i,3], label = md_geral[i,0], c=c[i])

plt.xlabel('Var 2', fontdict=font1)
plt.ylabel('Var 4', fontdict=font1)
plt.subplot(2,2,4)
for i in np.arange(0,10,1):
    plt.scatter(md_geral[i,1], md_geral[i,3], label = md_geral[i,0], c=c[i])

plt.legend(bbox_to_anchor=(1, 0.7), title='Genótipos', borderaxespad=0.5, ncol=1)

plt.show()

########################################################################################################################
########################################################################################################################
