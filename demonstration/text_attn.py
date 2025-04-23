# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-03-29 16:10:23
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-04-12 09:56:12


## convert the text/attention list to latex code, which will further generates the text heatmap based on attention weights.
import numpy as np

latex_special_token = ["!@#$%^&*()"]

def generate(idss, attn, tokenizer, color='red', rescale_value = False):
	assert(len(idss) == len(attn))
	if rescale_value:
		attention_list = rescale(attention_list)
	string = ""
	for idx in range(len(idss)):
		this_token = tokenizer.decode(idss[idx])
		# if this_token == "\n":
		# 	this_token = "\escape(n)"
		if this_token == ",":
			# print("hi")
			this_token ="\\text{ , }"
		string += "\colorbox{%s!%s}{"%(color, max(0.01, attn[idx]))+"\strut " + this_token +"} "
	# print("string=", string)
	return string

def rescale(input_list):
	the_array = np.asarray(input_list)
	the_max = np.max(the_array)
	the_min = np.min(the_array)
	rescale = (the_array - the_min)/(the_max-the_min)*100
	return rescale.tolist()


def clean_word(word_list):
	new_word_list = []
	for word in word_list:
		for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
			if latex_sensitive in word:
				word = word.replace(latex_sensitive, '\\'+latex_sensitive)
		new_word_list.append(word)
	return new_word_list


if __name__ == '__main__':
	## This is a demo:

	sent = '''the USS Ronald Reagan - an aircraft carrier docked in Japan - during his tour of the region, vowing to "defeat any attack and meet any use of conventional or nuclear weapons with an overwhelming and effective American response".
North Korea and the US have ratcheted up tensions in recent weeks and the movement of the strike group had raised the question of a pre-emptive strike by the US.
On Wednesday, Mr Pence described the country as the "most dangerous and urgent threat to peace and security" in the Asia-Pacific.'''
	words = sent.split()
	word_num = len(words)
	attention = [(x+1.)/word_num*100 for x in range(word_num)]
	import random
	random.seed(42)
	random.shuffle(attention)
	color = 'red'
	generate(words, attention, color)