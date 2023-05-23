# -*- coding: utf-8 -*-

import zlib
from math import exp

import string
import binascii
import re

import torch
from reedsolo import RSCodec
from torch.nn.functional import conv2d

rs = RSCodec(250)


def text_to_bits(text):
    
    """Convert text to a list of ints in {0, 1}"""
    #return bytearray_to_bits(text_to_bytearray(text))
    
    """Converts a string to a list of bits"""
    result = []
    for char in text:
        bits = bin(ord(char))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result



def bits_to_text(bits):
    
    """Convert a list of ints in {0, 1} to text"""
    return bytearray_to_text(bits_to_bytearray(bits))


def bytearray_to_bits(x):
    

    """Convert bytearray to a list of bits"""
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])

    
    return result

def bits_to_bytearray(bits):
    """Convert a list of bits to a bytearray"""
    
    result= []
    
    ints = []
    temp = []
    count=0
    for b in range(len(bits) // 8):
        temp = []

        byte = bits[b * 8:(b + 1) * 8]

        temp = ''.join(map(str,byte))

    
        ints = hex(int(temp,2))
        
        
        result.append(ints)


    my_list = result
    backslash_string = '\\'.join(my_list)

    return backslash_string

#def  bits_to_bytearray(bits):
   
    # Ensure length is a multiple of 8 bits (byte-aligned)
    padded_bits = bits + [0] * (len(bits) % 8)

    # Convert bits to binary string
    bin_str = ''.join([str(bit) for bit in padded_bits])

    # Split binary string into chunks of 8 bits (bytes), and convert each to integer byte
    bytes_arr = [int(bin_str[i:i+8], 2) for i in range(0, len(bin_str), 8)]

    # Encode bytes as string with \x format
    hex_str = ''.join([f'\\x{b:02x}' for b in bytes_arr])
    return bytes(hex_str.encode('utf-8'))


def text_to_bytearray(text):
    
    """Compress and add error correction"""
    assert isinstance(text, str), "expected a string"
    x = zlib.compress(text.encode("utf-8"))
    x = rs.encode(bytearray(x))
    
    return x

#def bytearray_to_text(byte_arr):
    """Converts a byte array with hex escape characters to a string"""
    hex_str = ''.join([chr(b) for b in byte_arr])
    return bytes.fromhex(hex_str).decode('utf-8')



#def bytearray_to_text(x):
    
    """Apply error correction and decompress"""
    try:
        text = rs.decode(x)
        text = zlib.decompress(text)
        return text.decode("utf-8")
    except BaseException:
        return False

def bytearray_to_text(hex_str):
    """Converts a byte array with hex escape characters to a string"""
    hex_str = '\\' + hex_str
    
    
    count=0
    result =''
    byte_list = []
    for sub_str in hex_str.split('\\'):
        if sub_str:
            count +=1
            sub_str=sub_str[1:]
            #print(sub_str)
            if len(sub_str) == 1:
                sub_str = '0' + sub_str

            text_value = bytearray.fromhex(sub_str).decode('latin-1')
            #print(text_value)

            '''
            if len(sub_str) == 1 and all(c in string.hexdigits for c in sub_str):
                sub_str = '0' + sub_str
                byte_list.append(int(sub_str, 16))
                if len(sub_str) == 2 and all(c in string.hexdigits for c in sub_str):
                    byte_list.append(int(sub_str, 16))
                else:
                    byte_list.extend(ord(x) for x in sub_str)
            '''

            byte_list.append(text_value)
            result += text_value
    
    #print(byte_string.decode('latin-1'))

    

    #regex_pattern = r"[^a-zA-Z0-9\s]+" # This pattern will match any characters that are NOT alphanumeric or whitespace

    #new_string = re.sub(regex_pattern, "", result)

    # Output: This is a sample string


    return result

def first_element(storage, loc):
    """Returns the first element of two"""
    return storage


def gaussian(window_size, sigma):
    """Gaussian window.

    https://en.wikipedia.org/wiki/Window_function#Gaussian_window
    """
    _exp = [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    gauss = torch.Tensor(_exp)
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):

    padding_size = window_size // 2

    mu1 = conv2d(img1, window, padding=padding_size, groups=channel)
    mu2 = conv2d(img2, window, padding=padding_size, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, padding=padding_size, groups=channel) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=padding_size, groups=channel) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, padding=padding_size, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    _ssim_quotient = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    _ssim_divident = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = _ssim_quotient / _ssim_divident

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
