/***************************************************************************
 *   Copyright (C) 2008-2022 by Walter Brisken & Chris Phillips            *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
//===========================================================================
// SVN properties (DO NOT CHANGE)
//
// $Id: m5spec.c 10609 2022-08-23 16:28:37Z WalterBrisken $
// $HeadURL: https://svn.atnf.csiro.au/difx/libraries/mark5access/trunk/mark5access/mark5_stream.c $
// $LastChangedRevision: 10609 $
// $Author: WalterBrisken $
// $LastChangedDate: 2022-08-24 02:28:37 +1000 (Wed, 24 Aug 2022) $
//
//============================================================================

// Change this to configure detection, if possible
#define USEGETOPT 1

#include "config.h"
#include <stdio.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>
#include <math.h>
#include <signal.h>
#include "../mark5access/mark5_stream.h"

#if USEGETOPT
#include <getopt.h>
#endif

const char program[] = "m5spec";
const char author[]  = "Walter Brisken, Chris Phillips";
const char version[] = "1.6";
const char verdate[] = "20220429";

volatile int die = 0;

typedef enum {VLBA=1, DBBC, NOPOL} polmodetype;

struct sigaction old_sigint_action;

void siginthand(int j)
{
	printf("\nBeing killed.  Partial results will be saved.\n\n");
	die = 1;

	sigaction(SIGINT, &old_sigint_action, 0);
}

static void usage(const char *pgm)
{
	printf("\n");

	printf("%s ver. %s   %s  %s\n\n", program, version, author, verdate);
	printf("A Mark5 spectrometer.  Can use VLBA, Mark3/4, Mark5B, and single-thread");
	printf("VDIF formats using the mark5access library.\n\n");
	printf("Usage : %s [options] <infile> <dataformat> <nchan> <nint> <outfile> [<offset>]\n\n", program);
	printf("  <infile> is the name of the input file\n\n");
	printf("  <dataformat> should be of the form: "
		"<FORMAT>-<Mbps>-<nchan>-<nbit>, e.g.:\n");
	printf("    VLBA1_2-256-8-2\n");
	printf("    MKIV1_4-128-2-1\n");
	printf("    Mark5B-512-16-2\n");
	printf("    VDIF_1000-64-1-2 (here 1000 is payload size in bytes)\n\n");
	printf("  alternatively for VDIF and CODIF, Mbps can be replaced by <FramesPerPeriod>m<AlignmentSeconds>, e.g.\n");
	printf("    VDIF_1000-8000m1-1-2 (8000 frames per 1 second, x1000 bytes x 8 bits= 64 Mbps)\n");
	printf("    CODIFC_5000-51200m27-8-1 (51200 frames every 27 seconds, x5000 bytes x 8 bits / 27 ~= 76 Mbps\n");
	printf("    This allows you to specify rates that are not an integer Mbps value, such as 32/27 CODIF oversampling\n\n");
	printf("  <nchan> is the number of spectral points to make per baseband channel\n\n");
	printf("  <nint> is the number of FFT frames to spectrometize; 0 -> run until end of file\n\n");
	printf("  <outfile> is the name of the output file\n\n");
	printf("  <offset> is number of bytes into file to start decoding\n\n");
	printf("\n");
	printf("The following options are supported\n\n");
	printf("    -version\n");
	printf("    -V         Print version info and quit\n\n");
	printf("    -dbbc\n");
	printf("    -B         Assume DBBC polarisation order (all Rcp then all Lcp)\n");
	printf("               Default is to assume Rcp/Lcp channel pairs\n\n");
	printf("    -nopol\n");
	printf("    -P         Do not compute cross pol terms\n\n");
	printf("    -double\n");
	printf("    -d         Double sideband (complex) data\n\n");
	printf("    -nonorm\n");
	printf("    -N         Do not normalise\n\n");
	printf("    -bchan=<x>\n");
	printf("    -b <x>     Start output at channel number <x> (0-based)\n\n");
	printf("    -echan=<x>\n");
	printf("    -e <x>     End output at channel number <x-1> (0-based)\n\n");
	printf("    -help\n");
	printf("    -h         Print this help info and quit\n\n");
}

int harvestComplexData(struct mark5_stream *ms, double **spec, fftw_complex **zdata, fftw_complex **zx, int nchan, int nint, int chunk, long long *total, long long *unpacked, int doublesideband)
{
	fftw_plan *plan;
	double complex **cdata;
	int j;

	plan = (fftw_plan *)malloc(ms->nchan*sizeof(fftw_plan));
	cdata = (double complex **)malloc(ms->nchan*sizeof(double complex *));
	for(j = 0; j < ms->nchan; ++j)
	{
		cdata[j] = (double complex*)malloc(nchan*sizeof(double complex));
		plan[j] = fftw_plan_dft_1d(nchan, cdata[j], zdata[j], FFTW_FORWARD, FFTW_MEASURE);
	}

	for(j = 0; j < nint; ++j)
	{
		int status;
		int i;

		if(die)
		{
			break;
		}

		status = mark5_stream_decode_double_complex(ms, chunk, (double complex**)cdata);
		if(status < 0)
		{
			break;
		}
		else
		{
			*total += chunk;
			*unpacked += status;
		}

		for(i = 0; i < ms->nchan; ++i)
		{
			/* FFT */
			fftw_execute(plan[i]);
		}

		for(i = 0; i < ms->nchan; ++i)
		{
			int c;

			for(c = 0; c < nchan; ++c)
			{
				double re, im;
				
				re = creal(zdata[i][c]);
				im = cimag(zdata[i][c]);
				spec[i][c] += re*re + im*im;
			}
		}

		for(i = 0; i < ms->nchan/2; ++i)
		{
			int c;

			for(c = 0; c < nchan; ++c)
			{
				zx[i][c] += zdata[2*i][c]*~zdata[2*i+1][c];
			}
		}
	}

	for(j = 0; j < ms->nchan; ++j)
	{
		fftw_destroy_plan(plan[j]);
		free(cdata[j]);
	}
	free(plan);
	free(cdata);

	fftw_cleanup();

	// If Double sideband need to move stuff around

	if(doublesideband) 
	{
		int i;
		double dtmp;
		fftw_complex ctmp;
		for(i = 0; i < ms->nchan; ++i)
		{
			int c;

			for(c = 0; c < nchan/2; ++c)
			{
				dtmp = spec[i][c];
				spec[i][c] = spec[i][c+nchan/2];
				spec[i][c+nchan/2] = dtmp;
			}
		}
		for(i = 0; i < ms->nchan/2; ++i)
		{
			int c;

			for(c = 0; c < nchan/2; ++c)
			{
				ctmp = zx[i][c];
				zx[i][c] = zx[i][c+nchan/2];
				zx[i][c+nchan/2] = ctmp;
			}
		}
	}

	return 0;
}

int harvestRealData(struct mark5_stream *ms, double **spec, fftw_complex **zdata, fftw_complex **zx, int nchan, int nint, int chunk, long long *total, long long *unpacked, polmodetype polmode)
{
	fftw_plan *plan;
	double **data;
	int j;

	plan = (fftw_plan *)malloc(ms->nchan*sizeof(fftw_plan));
	data = (double **)malloc(ms->nchan*sizeof(double *));
	for(j = 0; j < ms->nchan; ++j)
	{
		data[j] = (double *)malloc((chunk+2)*sizeof(double));
		plan[j] = fftw_plan_dft_r2c_1d(nchan*2, data[j], zdata[j], FFTW_MEASURE);
	}
	for(j = 0; j < nint; ++j)
	{
		int status;
		int i;

		if(die)
		{
			break;
		}

		status = mark5_stream_decode_double(ms, chunk, data);
		if(status < 0)
		{
			break;
		}
		else
		{
			*total += chunk;
			*unpacked += status;
		}

		for(i = 0; i < ms->nchan; ++i)
		{
			/* FFT */
			fftw_execute(plan[i]);
		}

		for(i = 0; i < ms->nchan; ++i)
		{
			int c;

			for(c = 0; c < nchan; ++c)
			{
				double re, im;
				
				re = creal(zdata[i][c]);
				im = cimag(zdata[i][c]);
				spec[i][c] += re*re + im*im;
			}
		}

		if(polmode==VLBA) 
		{
			for(i = 0; i < ms->nchan/2; ++i)
			{
				int c;

				for(c = 0; c < nchan; ++c)
				{
					zx[i][c] += zdata[2*i][c]*~zdata[2*i+1][c];
				}
			}
		} 
		else if(polmode==DBBC) 
		{
			for(i = 0; i < ms->nchan/2; ++i)
			{
				int c;

				for(c = 0; c < nchan; ++c)
				{
					zx[i][c] += zdata[i][c]*~zdata[i+ms->nchan/2][c];
				}
			}
		}
	}
	for(j = 0; j < ms->nchan; ++j)
	{
		fftw_destroy_plan(plan[j]);
		free(data[j]);
	}
	free(plan);
	free(data);

	fftw_cleanup();

	return 0;
}


int spec(const char *filename, const char *formatname, int nchan, int nint, const char *outfile, long long offset,
	 polmodetype polmode, int doublesideband, int nonorm, int bchan, int echan)
{
	struct mark5_stream *ms;
	double **spec;
	fftw_complex **zdata, **zx;
	int i, c;
	int chunk;
	long long total, unpacked;
	FILE *out;
	double f, sum, chanbw;
	double x, y;
	int docomplex;

	total = unpacked = 0;

	ms = new_mark5_stream_absorb(
		new_mark5_stream_file(filename, offset),
		new_mark5_format_generic_from_string(formatname) );

	if(!ms)
	{
		printf("Error: problem opening %s\n", filename);

		return EXIT_FAILURE;
	}

	mark5_stream_print(ms);

	if(ms->iscomplex) 
	{
		printf("Complex decode\n");
		docomplex = 1;
		chunk = nchan;
	}
	else
	{
		docomplex = 0;
		if(doublesideband)
		{
			printf("Warning Double sideband supported only for complex sampled data\n");
			doublesideband = 0;
		}
		chunk = 2*nchan;
	}

	out = fopen(outfile, "w");
	if(!out)
	{
		fprintf(stderr, "Error: cannot open %s for write\n", outfile);
		delete_mark5_stream(ms);

		return EXIT_FAILURE;
	}

	spec = (double **)malloc(ms->nchan*sizeof(double *));
	zdata = (fftw_complex **)malloc(ms->nchan*sizeof(fftw_complex *));
	zx = (fftw_complex **)malloc((ms->nchan/2)*sizeof(fftw_complex *));
	for(i = 0; i < ms->nchan; ++i)
	{
		spec[i] = (double *)calloc(nchan, sizeof(double));
		zdata[i] = (fftw_complex *)malloc((nchan+2)*sizeof(fftw_complex));
	}
	for(i = 0; i < ms->nchan/2; ++i)
	{
		zx[i] = (fftw_complex *)calloc(nchan, sizeof(fftw_complex));
	}

	if(docomplex)
	{
		harvestComplexData(ms, spec, zdata, zx, nchan, nint, chunk, &total, &unpacked, doublesideband);
	} 
	else
	{
		harvestRealData(ms, spec, zdata, zx, nchan, nint, chunk, &total, &unpacked, polmode);
	}

	fprintf(stderr, "%lld / %lld samples unpacked\n", unpacked, total);

	/* normalize across all ifs/channels */
	sum = 0.0;
	for(c = 0; c < nchan; ++c)
	{
		for(i = 0; i < ms->nchan; ++i)
		{
			sum += spec[i][c];
		}
	}

	f = ms->nchan*nchan/sum;
	if(nonorm)
	{
		//printf("Norm Factor = %.3g\n", 1/f);
		//f *= unpacked*256;
		//printf("Updated Norm Factor = %.3g\n", 1/f);
		f = 1.0/unpacked/256.0;
	}
	
	chanbw = ms->samprate/(2.0e6*nchan);
	if(docomplex)
	{
		chanbw *= 2;
	}

	for(c = bchan; c < echan; ++c)
	{
		fprintf(out, "%f ", (double)c*chanbw);
		for(i = 0; i < ms->nchan; ++i)
		{
			fprintf(out, " %f", f*spec[i][c]);
		}
		if(polmode != NOPOL)
		{
		        for(i = 0; i < ms->nchan/2; ++i)
			{
			        x = creal(zx[i][c])*f;
				y = cimag(zx[i][c])*f;
				fprintf(out, "  %f %f", sqrt(x*x+y*y), atan2(y, x));
			}
		}
		fprintf(out, "\n");
	}

	fclose(out);

	for(i = 0; i < ms->nchan; ++i)
	{
		free(zdata[i]);
		free(spec[i]);
	}
	for(i = 0; i < ms->nchan/2; ++i)
	{
		free(zx[i]);
	}
	free(zx);
	free(zdata);
	free(spec);
	delete_mark5_stream(ms);

	return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
	long long offset = 0;
	int nchan, nint;
	int bchan=-1, echan=-1;
	int retval;
	polmodetype polmode = VLBA;
	int doublesideband = 0;
	int nonorm = 0;
	struct sigaction new_sigint_action;
#if USEGETOPT
	int opt;
	struct option options[] = {
		{"version", 0, 0, 'V'}, // Version
		{"double", 0, 0, 'd'}, // Double sideband complex
		{"nonorm", 0, 0, 'N'}, // Don't normalise
		{"dbbc", 0, 0, 'B'},  // DBBC channel ordering
		{"nopol", 0, 0, 'P'}, // Don't compute the crosspol terms
		{"help", 0, 0, 'h'},
		{"bchan", 1, 0, 'b'},
		{"echan", 1, 0, 'e'},
		{0, 0, 0, 0}
	};

	while((opt = getopt_long_only(argc, argv, "NVdBPhb:e:", options, NULL)) != EOF)
	{
		switch (opt) 
		{
		case 'B': // DBBC Pol mode (all Rcp then all LCP)
			polmode = DBBC;
			printf("Assuming DBBC polarisation order\n");
			break;

		case 'P': // Don't compute cross pols
			polmode = NOPOL;
			printf("Not computing cross pol terms\n");
			break;

		case 'd': // Double sideband
			doublesideband = 1;
			printf("Assuming double sideband data\n");
			break;

		case 'N': // No normalisation
			nonorm = 1;
			printf("Not normalising\n");
			break;

		case 'V': // Version
			printf("%s ver. %s   %s  %s\n\n", program, version, author, verdate);
			return EXIT_SUCCESS;

		case 'b': // bchan (output selection)
			bchan = atoi(optarg);
			break;

		case 'e': // echan (output selection)
			echan = atoi(optarg);
			break;

		case 'h': // help
			usage(argv[0]);
			return EXIT_SUCCESS;
		}
	}

#else
	int optind=1;
#endif

	if(argc-optind == 1)
	{
		struct mark5_format *mf;
		int bufferlen = 1<<11;
		char *buffer;
		FILE *in;
		int r;

		if(strcmp(argv[1], "-h") == 0 ||
		   strcmp(argv[1], "--help") == 0)
		{
			usage(argv[0]);

			return EXIT_SUCCESS;
		}

		in = fopen(argv[1], "r");
		if(!in)
		{
			fprintf(stderr, "Error: cannot open file '%s'\n", argv[1]);
			
			return EXIT_FAILURE;
		}

		buffer = malloc(bufferlen);

		r = fread(buffer, 1, bufferlen, in);
		if(r < bufferlen)
		{
			fprintf(stderr, "Error: cannot read %d bytes from file\n", bufferlen);
			fprintf(stderr, "Error:   just read %d bytes from file\n", r);
			fclose(in);
			free(buffer);
			return -1;
		}
		else
		{
			mf = new_mark5_format_from_stream(new_mark5_stream_memory(buffer, bufferlen/2));

			print_mark5_format(mf);
			delete_mark5_format(mf);

			mf = new_mark5_format_from_stream(new_mark5_stream_memory(buffer, bufferlen/2));

			print_mark5_format(mf);
			delete_mark5_format(mf);
		}

		fclose(in);
		free(buffer);

		return EXIT_SUCCESS;
	}

	else if(argc-optind < 5)
	{
		usage(argv[0]);

		return EXIT_FAILURE;
	}

	nchan = atol(argv[optind+2]);
	nint  = atol(argv[optind+3]);
	if(nint <= 0)
	{
		nint = 2000000000L;
	}

	if(argc-optind > 5)
	{
		offset = atoll(argv[optind+5]);
	}

	new_sigint_action.sa_handler = siginthand;
	sigemptyset(&new_sigint_action.sa_mask);
	new_sigint_action.sa_flags = 0;
	sigaction(SIGINT, &new_sigint_action, &old_sigint_action);

	if(bchan >= 0 && echan >= 0 && bchan >= echan)
	{
		fprintf(stderr, "Error: if bchan and echan are provided, echan must exceed bchan\n");

		return EXIT_FAILURE;
	}

	if(bchan < 0)
	{
		bchan = 0;
	}
	if(echan > nchan || echan < 0)
	{
		echan = nchan;
	}

	retval = spec(argv[optind], argv[optind+1], nchan, nint, argv[optind+4], offset, polmode, doublesideband, nonorm, bchan, echan);

	return retval;
}

